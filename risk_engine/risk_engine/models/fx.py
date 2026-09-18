"""
FX spot model family: FXGBM, FXGBM_SV -- for USDJPY (or any base/quote pair
with a compo trade in the book).

dF/F = (r_usd(t) - r_jpy(t)) dt + sigma(t) dW          [FXGBM]
dF/F = (r_usd(t) - r_jpy(t)) dt + sigma(t)*sqrt(v(t)) dW  [FXGBM_SV]

r_usd(t) is read off the SAME simulated path's rate state via the paired
domestic CalibratedRateModel, exactly as in models/gbm.py.

r_jpy(t) is now ALSO a genuinely simulated rate factor (a real JPY LGM
model, jointly simulated with USD and every other factor via the same
Cholesky correlation structure -- see simulation/joint.py), calibrated off
the real Bloomberg JPY OIS curve + swaption vol cube
(market_data/bloomberg_data.py), NOT a deterministic implied-from-forwards
curve as this module used before that real JPY data existed (see git
history / the module's earlier docstring for the prior design: a static
covered-interest-parity-implied curve via
risk_engine.calibration.implied_fx_curve, used only because "there is no
JPY rate factor in this book"). foreign_rate_model below plays exactly the
domestic rate_model's role, just for the quote currency -- this book still
has no JPY-denominated CASHFLOW (the JPY equity basket is a USD-settled FX
compo, not JPY discounting), so the JPY factor's ONLY consumer is this
FX drift; it is not used to discount any trade directly.

A deterministic implied_foreign_curve fallback path is still supported
(foreign_rate_model=None, implied_foreign_curve=<Curve>) for callers that
have not been updated to source a real JPY rate model -- this keeps
existing callers/tests working without a breaking change, while making the
real simulated-factor path the preferred one going forward.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import SpotModel, CalibratedSpotModel
from ._shared import (
    bootstrap_sigma, PiecewiseSigma, fit_theta, fit_skew_smile, year_frac,
    simulate_cir_variance_step, leveraged_vol_shock,
)
from ..calibration import priors as _priors


@dataclass
class _CalibratedFXGBM(CalibratedSpotModel):
    spot: float
    ref_date: object
    rate_model: object          # CalibratedRateModel, domestic (USD) leg
    implied_foreign_curve: object = None   # deterministic Curve fallback, foreign (JPY) leg --
                                            # used ONLY when foreign_rate_model is None
    foreign_rate_model: object = None      # OPTIONAL real CalibratedRateModel, foreign (JPY) leg --
                                            # when given, this is the PREFERRED path (see module
                                            # docstring); implied_foreign_curve is then unused
    sigma: PiecewiseSigma = None
    sv: Optional[_priors.SVPriors] = None
    assumptions: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.foreign_rate_model is None and self.implied_foreign_curve is None:
            raise ValueError("_CalibratedFXGBM needs EITHER foreign_rate_model (a real simulated "
                              "foreign-currency CalibratedRateModel) OR implied_foreign_curve "
                              "(the deterministic covered-interest-parity fallback) -- neither was given.")

    def zeta(self, t: float) -> float:
        return self.sigma.zeta(t)

    def _foreign_df(self, t: float, foreign_rate_state=None) -> float:
        if self.foreign_rate_model is not None:
            if foreign_rate_state is None:
                raise ValueError("FXGBM.level_at requires foreign_rate_state when calibrated with a "
                                  "real foreign_rate_model (see models/fx.py's module docstring) -- "
                                  "the caller (scenario_market.build_market_states_at) must pass the "
                                  "foreign factor's own simulated state, not just the domestic one.")
            return self.foreign_rate_model.discount_factor(foreign_rate_state, 0.0, t) if t > 0 else 1.0
        return self.implied_foreign_curve.discount(_shift(self.ref_date, t)) if t > 0 else 1.0

    def level_at(self, state, t: float, rate_state=None, foreign_rate_state=None) -> float:
        if rate_state is None:
            raise ValueError("FXGBM.level_at requires rate_state (the paired USD rate model's "
                              "simulated state at t) -- domestic drift is read off that state.")
        log_growth, zeta = state[0], state[1]
        df_usd_0_t = self.rate_model.discount_factor(rate_state, 0.0, t) if t > 0 else 1.0
        df_jpy_0_t = self._foreign_df(t, foreign_rate_state)
        # F(t) domestic-measure drift: growth at r_usd, discounted back by the
        # foreign leg (real simulated JPY factor if given, else the
        # deterministic implied curve) -- same construction as GBM's
        # dividend yield, with df_jpy_0_t playing the role of exp(-q*t).
        return self.spot * (df_jpy_0_t / df_usd_0_t) * np.exp(-0.5 * zeta + log_growth)

    def level_at_batch(self, state_batch: np.ndarray, t: float, rate_state_batch: np.ndarray,
                        foreign_rate_state_batch: np.ndarray = None) -> np.ndarray:
        """Vectorized level_at -- see GBM.level_at_batch's docstring for why
        this exists (build_market_states_at's per-path Python loop dominates
        pricing wall-clock at scale)."""
        log_growth, zeta = state_batch[:, 0], state_batch[:, 1]
        n_paths = state_batch.shape[0]
        if t > 0:
            df_usd_0_t = self.rate_model.discount_factor_batch(rate_state_batch, 0.0, t)
            if self.foreign_rate_model is not None:
                if foreign_rate_state_batch is None:
                    raise ValueError("FXGBM.level_at_batch requires foreign_rate_state_batch when "
                                      "calibrated with a real foreign_rate_model.")
                df_jpy_0_t = self.foreign_rate_model.discount_factor_batch(foreign_rate_state_batch, 0.0, t)
            else:
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
                vz = leveraged_vol_shock(z, self.sv.rho, rng, n_paths)
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

    def calibrate(self, spot, drift_source, vol_surface, implied_foreign_curve=None,
                  foreign_rate_model=None, **kw) -> _CalibratedFXGBM:
        """drift_source: the paired CalibratedRateModel (USD, domestic).

        foreign_rate_model: PREFERRED -- a real, already-calibrated foreign-
            currency CalibratedRateModel (e.g. a JPY LGM model calibrated
            off the real Bloomberg JPY OIS curve + swaption vol, see
            market_data/bloomberg_data.py), jointly simulated alongside the
            domestic factor. When given, implied_foreign_curve is ignored.

        implied_foreign_curve: FALLBACK ONLY, used when foreign_rate_model
            is not given -- deterministic JPY curve from
            risk_engine.calibration.implied_fx_curve.build_implied_jpy_curve
            (or flat_implied_jpy_curve). Kept for callers that have not
            been updated to source a real foreign rate model.
        """
        if foreign_rate_model is None and implied_foreign_curve is None:
            raise ValueError("FXGBM.calibrate requires EITHER foreign_rate_model (preferred -- a real "
                              "calibrated foreign-currency rate model) OR implied_foreign_curve (the "
                              "deterministic covered-interest-parity fallback)")
        pillars, sigmas = bootstrap_sigma(vol_surface.atm_term_structure())
        return _CalibratedFXGBM(
            spot=spot, ref_date=drift_source.curve.ref_date, rate_model=drift_source,
            implied_foreign_curve=implied_foreign_curve, foreign_rate_model=foreign_rate_model,
            sigma=PiecewiseSigma(pillars, sigmas),
            assumptions={"sigma_source": "ATM term vol, bootstrapped",
                         "foreign_drift_source": ("real simulated foreign rate factor (Bloomberg JPY OIS + swaption vol)"
                                                   if foreign_rate_model is not None
                                                   else "implied JPY curve via covered interest parity")},
        )


class FXGBM_SV(SpotModel):
    name = "FXGBM_SV"

    def __init__(self, sv_prior: _priors.SVPriors = None):
        self.sv_prior = sv_prior

    def calibrate(self, spot, drift_source, vol_surface, implied_foreign_curve=None,
                  foreign_rate_model=None, **kw) -> _CalibratedFXGBM:
        """See FXGBM.calibrate's docstring -- foreign_rate_model (preferred,
        a real calibrated foreign-currency rate model) vs.
        implied_foreign_curve (deterministic fallback) apply identically here."""
        if foreign_rate_model is None and implied_foreign_curve is None:
            raise ValueError("FXGBM_SV.calibrate requires EITHER foreign_rate_model (preferred) OR "
                              "implied_foreign_curve (fallback) -- see FXGBM.calibrate's docstring")
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
            implied_foreign_curve=implied_foreign_curve, foreign_rate_model=foreign_rate_model,
            sigma=PiecewiseSigma(pillars, sigmas), sv=sv,
            assumptions={
                "sigma_source": "ATM term vol, bootstrapped",
                "foreign_drift_source": ("real simulated foreign rate factor (Bloomberg JPY OIS + swaption vol)"
                                          if foreign_rate_model is not None
                                          else "implied JPY curve via covered interest parity"),
                "theta_source": "ATM term vol level (identifiable)",
                "eta_source": "smile curvature" if eta_smile is not None else "literature prior (surface under-determined)",
                "rho_source": "smile skew" if rho_smile is not None else "literature prior (surface under-determined)",
                "kappa_source": "literature prior (no FX-option vol-of-vol grid to identify mean-reversion of variance)",
            },
        )
