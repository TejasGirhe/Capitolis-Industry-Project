"""
FX spot model family: FXGBM, FXGBM_SV -- for USDJPY (or any base/quote pair
with a compo trade in the book).

dF/F = (r_usd(t) - r_jpy_implied(t)) dt + sigma(t) dW          [FXGBM]
dF/F = (r_usd(t) - r_jpy_implied(t)) dt + sigma(t)*sqrt(v(t)) dW  [FXGBM_SV]

r_usd(t) is read off the SAME simulated path's rate state via the paired
CalibratedRateModel, exactly as in models/gbm.py. r_jpy_implied(t) is
deterministic -- backed out once from the FX forward curve via covered
interest parity (risk_engine.calibration.implied_fx_curve), since there is no
JPY rate factor in this book (MARKET_DATA.md Sec.2.2). Structurally identical
to GBM otherwise; kept as a separate module (not a GBM subclass) because the
two-curve drift (domestic minus deterministic-implied-foreign) is a distinct
calibration contract from GBM's single dividend-yield drift, even though the
diffusion/state mechanics are the same.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import SpotModel, CalibratedSpotModel
from ._shared import bootstrap_sigma, PiecewiseSigma, fit_theta, fit_skew_smile, year_frac, simulate_cir_variance_step
from ..calibration import priors as _priors


@dataclass
class _CalibratedFXGBM(CalibratedSpotModel):
    spot: float
    ref_date: object
    rate_model: object          # CalibratedRateModel, domestic (USD) leg
    implied_foreign_curve: object   # deterministic Curve, foreign (JPY) leg
    sigma: PiecewiseSigma
    sv: Optional[_priors.SVPriors] = None
    assumptions: dict = field(default_factory=dict)

    def zeta(self, t: float) -> float:
        return self.sigma.zeta(t)

    def level_at(self, state, t: float, rate_state=None) -> float:
        if rate_state is None:
            raise ValueError("FXGBM.level_at requires rate_state (the paired USD rate model's "
                              "simulated state at t) -- domestic drift is read off that state.")
        log_growth, zeta = state[0], state[1]
        df_usd_0_t = self.rate_model.discount_factor(rate_state, 0.0, t) if t > 0 else 1.0
        df_jpy_0_t = self.implied_foreign_curve.discount(_shift(self.ref_date, t)) if t > 0 else 1.0
        # F(t) domestic-measure drift: growth at r_usd, discounted back by the
        # deterministic foreign curve -- same construction as GBM's dividend
        # yield, with df_jpy_0_t playing the role of exp(-q*t).
        return self.spot * (df_jpy_0_t / df_usd_0_t) * np.exp(-0.5 * zeta + log_growth)

    def level_at_batch(self, state_batch: np.ndarray, t: float, rate_state_batch: np.ndarray) -> np.ndarray:
        """Vectorized level_at -- see GBM.level_at_batch's docstring for why
        this exists (build_market_states_at's per-path Python loop dominates
        pricing wall-clock at scale)."""
        log_growth, zeta = state_batch[:, 0], state_batch[:, 1]
        n_paths = state_batch.shape[0]
        if t > 0:
            df_usd_0_t = self.rate_model.discount_factor_batch(rate_state_batch, 0.0, t)
            df_jpy_0_t = self.implied_foreign_curve.discount(_shift(self.ref_date, t)) * np.ones(n_paths)
        else:
            df_usd_0_t = np.ones(n_paths)
            df_jpy_0_t = np.ones(n_paths)
        return self.spot * (df_jpy_0_t / df_usd_0_t) * np.exp(-0.5 * zeta + log_growth)

    def simulate_paths(self, n_paths: int, horizon_dates, rng, external_z=None):
        times = [year_frac(self.ref_date, d) for d in horizon_dates]
        times = [0.0] + times
        n_steps = len(times) - 1
        n_dates = len(horizon_dates)
        has_sv = self.sv is not None

        log_growth = np.zeros((n_paths, n_steps + 1))
        det_zeta = np.zeros(n_steps + 1)
        v = np.ones((n_paths, n_steps + 1))

        for step in range(n_steps):
            t0, t1 = times[step], times[step + 1]
            if t1 <= t0:
                # No time elapsed -- see models/gbm.py's simulate_paths for
                # the full rationale (same bug pattern, same fix).
                log_growth[:, step + 1] = log_growth[:, step]
                det_zeta[step + 1] = det_zeta[step]
                v[:, step + 1] = v[:, step]
                continue
            dt = t1 - t0
            z = external_z[:, step, 0] if external_z is not None else rng.standard_normal(n_paths)
            sigma_t = self.sigma.at(t0)
            if has_sv:
                vz = rng.standard_normal(n_paths)
                v[:, step + 1] = simulate_cir_variance_step(v[:, step], self.sv.kappa, self.sv.eta, dt, vz)
                eff_sigma = sigma_t * np.sqrt(np.maximum(v[:, step], 0.0))
            else:
                eff_sigma = sigma_t
                det_zeta[step + 1] = det_zeta[step] + sigma_t ** 2 * dt
            log_growth[:, step + 1] = (log_growth[:, step] - 0.5 * eff_sigma ** 2 * dt
                                        + eff_sigma * np.sqrt(dt) * z)

        out = np.zeros((n_paths, n_dates, 2))
        for i in range(n_dates):
            step_idx = i + 1
            out[:, i, 0] = log_growth[:, step_idx]
            out[:, i, 1] = det_zeta[step_idx]
        return out


def _shift(ref_date, years):
    from datetime import timedelta
    return ref_date + timedelta(days=round(years * 365.0))


class FXGBM(SpotModel):
    name = "FXGBM"

    def calibrate(self, spot, drift_source, vol_surface, implied_foreign_curve=None, **kw) -> _CalibratedFXGBM:
        """drift_source: the paired CalibratedRateModel (USD, domestic).
        implied_foreign_curve: deterministic JPY curve from
            risk_engine.calibration.implied_fx_curve.build_implied_jpy_curve
            (or flat_implied_jpy_curve as a documented fallback)."""
        if implied_foreign_curve is None:
            raise ValueError("FXGBM.calibrate requires implied_foreign_curve -- build it via "
                              "risk_engine.calibration.implied_fx_curve.build_implied_jpy_curve")
        pillars, sigmas = bootstrap_sigma(vol_surface.atm_term_structure())
        return _CalibratedFXGBM(
            spot=spot, ref_date=drift_source.curve.ref_date, rate_model=drift_source,
            implied_foreign_curve=implied_foreign_curve, sigma=PiecewiseSigma(pillars, sigmas),
            assumptions={"sigma_source": "ATM term vol, bootstrapped",
                         "foreign_drift_source": "implied JPY curve via covered interest parity"},
        )


class FXGBM_SV(SpotModel):
    name = "FXGBM_SV"

    def __init__(self, sv_prior: _priors.SVPriors = None):
        self.sv_prior = sv_prior

    def calibrate(self, spot, drift_source, vol_surface, implied_foreign_curve=None, **kw) -> _CalibratedFXGBM:
        if implied_foreign_curve is None:
            raise ValueError("FXGBM_SV.calibrate requires implied_foreign_curve -- build it via "
                              "risk_engine.calibration.implied_fx_curve.build_implied_jpy_curve")
        pillars, sigmas = bootstrap_sigma(vol_surface.atm_term_structure())
        theta = fit_theta(vol_surface)
        prior = self.sv_prior or _priors.FX_SV_PRIOR
        rho_smile, eta_smile = fit_skew_smile(vol_surface)
        sv = _priors.SVPriors(
            kappa=prior.kappa, theta=theta,
            eta=eta_smile if eta_smile is not None else prior.eta,
            rho=rho_smile if rho_smile is not None else prior.rho,
        )
        return _CalibratedFXGBM(
            spot=spot, ref_date=drift_source.curve.ref_date, rate_model=drift_source,
            implied_foreign_curve=implied_foreign_curve, sigma=PiecewiseSigma(pillars, sigmas), sv=sv,
            assumptions={
                "sigma_source": "ATM term vol, bootstrapped",
                "foreign_drift_source": "implied JPY curve via covered interest parity",
                "theta_source": "ATM term vol level (identifiable)",
                "eta_source": "smile curvature" if eta_smile is not None else "literature prior (surface under-determined)",
                "rho_source": "smile skew" if rho_smile is not None else "literature prior (surface under-determined)",
                "kappa_source": "literature prior (no FX-option vol-of-vol grid to identify mean-reversion of variance)",
            },
        )
