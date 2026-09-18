"""
Equity spot model family: GBM, GBM_SV.

dS/S = (r(t) - q) dt + sigma(t) dW      [GBM]
dS/S = (r(t) - q) dt + sigma(t)*sqrt(v(t)) dW,  dv = kappa(1-v)dt + eta*sqrt(v)dWv   [GBM_SV]

r(t) is NOT a static curve here -- it is read off the SAME simulated path's
short-rate integral from the paired CalibratedRateModel (via
discount_factor(), see models/lgm.py), so equity drift is path-consistent
with whatever rate scenario that path represents: two paths with different
realized rates get different equity drifts, not a shared average curve. This
is the "maximum accuracy" requirement from calibration.

State returned by simulate_paths is a plain (n_paths, n_dates, 2) float array
of [log_growth, zeta_or_zero] per path/date -- NOT the level itself:
    log_growth accumulates  -0.5*eff_sigma(s)^2 ds + eff_sigma(s) dW(s)
    (the pure diffusion martingale correction + noise; eff_sigma already
    includes the CIR multiplier sqrt(v(s)) for the -SV variant, so no
    separate "zeta" term is needed then -- zeta_or_zero is only the
    deterministic-vol zeta(t), used by level_at for the non-SV case, and
    left at 0 for -SV since it's already folded into log_growth).

level_at(state, t, rate_state) combines this with the paired rate model's
discount_factor(rate_state, 0, t) to get the exact per-path forward:
    S_p(t) = S(0) / DF_p(0,t) * exp(-q*t + log_growth_p(t))
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
class _CalibratedGBM(CalibratedSpotModel):
    spot: float
    dividend_rate: float
    ref_date: object
    rate_model: object   # CalibratedRateModel, for path-consistent drift
    sigma: PiecewiseSigma
    sv: Optional[_priors.SVPriors] = None
    assumptions: dict = field(default_factory=dict)

    def zeta(self, t: float) -> float:
        return self.sigma.zeta(t)

    def level_at(self, state, t: float, rate_state=None) -> float:
        if rate_state is None:
            raise ValueError("GBM.level_at requires rate_state (the paired rate model's simulated "
                              "state at t) -- equity drift is read off that state, see module docstring.")
        log_growth, zeta = state[0], state[1]
        df_p_0_t = self.rate_model.discount_factor(rate_state, 0.0, t) if t > 0 else 1.0
        return self.spot * (1.0 / df_p_0_t) * np.exp(-self.dividend_rate * t - 0.5 * zeta + log_growth)

    def level_at_batch(self, state_batch: np.ndarray, t: float, rate_state_batch: np.ndarray) -> np.ndarray:
        """Vectorized level_at: state_batch/rate_state_batch shape
        (n_paths, n_state_vars) -> (n_paths,) levels. Used by
        build_market_states_at, which otherwise calls level_at once per path
        per regression date and dominates pricing wall-clock at scale (37
        equity factors x ~100 regression dates x n_paths x 2 curves)."""
        log_growth, zeta = state_batch[:, 0], state_batch[:, 1]
        if t > 0:
            df_p_0_t = self.rate_model.discount_factor_batch(rate_state_batch, 0.0, t)
        else:
            df_p_0_t = np.ones(state_batch.shape[0])
        return self.spot * (1.0 / df_p_0_t) * np.exp(-self.dividend_rate * t - 0.5 * zeta + log_growth)

    def simulate_paths(self, n_paths: int, horizon_dates, rng, external_z=None):
        times = [year_frac(self.ref_date, d) for d in horizon_dates]
        times = [0.0] + times
        n_steps = len(times) - 1
        n_dates = len(horizon_dates)
        has_sv = self.sv is not None

        log_growth = np.zeros((n_paths, n_steps + 1))
        det_zeta = np.zeros(n_steps + 1)   # deterministic-vol zeta(t); unused (left 0) when has_sv
        v = np.ones((n_paths, n_steps + 1))

        for step in range(n_steps):
            t0, t1 = times[step], times[step + 1]
            if t1 <= t0:
                # No time elapsed (e.g. horizon_dates' first entry is
                # ref_date itself) -- copy state forward with zero diffusion
                # rather than stepping over a spurious dt floor, which would
                # inject noise into what should be an exact zero state at
                # t=0 (see models/lgm.py's simulate_paths for the full
                # rationale; same bug pattern, same fix).
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
            out[:, i, 1] = det_zeta[step_idx]  # 0.0 for every date when has_sv
        return out


class GBM(SpotModel):
    name = "GBM"

    def calibrate(self, spot, drift_source, vol_surface, dividend_rate=0.0, **kw) -> _CalibratedGBM:
        """drift_source: the paired CalibratedRateModel (USD)."""
        pillars, sigmas = bootstrap_sigma(vol_surface.atm_term_structure())
        return _CalibratedGBM(
            spot=spot, dividend_rate=dividend_rate, ref_date=drift_source.curve.ref_date,
            rate_model=drift_source, sigma=PiecewiseSigma(pillars, sigmas),
            assumptions={"sigma_source": "ATM term vol, bootstrapped", "dividend_rate": dividend_rate},
        )


class GBM_SV(SpotModel):
    name = "GBM_SV"

    def __init__(self, sv_prior: _priors.SVPriors = None):
        self.sv_prior = sv_prior

    def calibrate(self, spot, drift_source, vol_surface, dividend_rate=0.0, **kw) -> _CalibratedGBM:
        pillars, sigmas = bootstrap_sigma(vol_surface.atm_term_structure())
        theta = fit_theta(vol_surface)
        prior = self.sv_prior or _priors.EQUITY_SV_PRIOR
        rho_smile, eta_smile = fit_skew_smile(vol_surface)
        sv = _priors.SVPriors(
            kappa=prior.kappa, theta=theta,
            eta=eta_smile if eta_smile is not None else prior.eta,
            rho=rho_smile if rho_smile is not None else prior.rho,
        )
        return _CalibratedGBM(
            spot=spot, dividend_rate=dividend_rate, ref_date=drift_source.curve.ref_date,
            rate_model=drift_source, sigma=PiecewiseSigma(pillars, sigmas), sv=sv,
            assumptions={
                "sigma_source": "ATM term vol, bootstrapped",
                "dividend_rate": dividend_rate,
                "theta_source": "ATM term vol level (identifiable)",
                "eta_source": "smile curvature" if eta_smile is not None else "literature prior (surface under-determined)",
                "rho_source": "smile skew" if rho_smile is not None else "literature prior (surface under-determined)",
                "kappa_source": "literature prior (no equity-option vol-of-vol grid to identify mean-reversion of variance)",
            },
        )
