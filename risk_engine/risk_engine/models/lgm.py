"""
Linear Gauss-Markov (LGM) short-rate model family: LGM1F, LGM1F_SV, LGM2F, LGM2F_SV.

LGM is the Hull-White short-rate model written in the "linear Gauss-Markov"
/ separable-HJM form used by QuantLib/ORE (the lineage capitolis_pricers'
DCF conventions follow). State variable(s) x(t) are driftless Gaussian
(Brownian, scaled by sigma(t)) under the T-forward / LGM numeraire measure;
all curve dependence sits in the deterministic shift function H(t) and in the
DF(0,.) terms of the zero-coupon-bond formula below -- so forward/curve
matching is exact by construction, not an approximation fitted after the
fact:

    DF(t,T) = DF(0,T)/DF(0,t) * exp( -H(T-t) x(t) - 0.5 H(T-t)^2 zeta(t) )

where zeta(t) = integral_0^t sigma(s)^2 ds is the accumulated variance and
H(u) = (1 - exp(-a u)) / a is the mean-reversion shift (a = mean reversion
speed). Taking expectations under the model's own T-forward measure reproduces
DF(0,T)/DF(0,t) exactly -- this identity is what "forward matching" means for
an LGM model, and it is what test_lgm_forward_matching.py checks numerically
by Monte Carlo averaging.

2-factor variants add a second state variable x2(t) with its own (a2, sigma2)
and a correlation rho_12 between the driving Brownians; the DF formula sums
both factors' H_i * x_i and 0.5 H_i^2 zeta_i contributions plus a cross term.

Stochastic-vol (-SV) variants multiply each factor's diffusion by a CIR
variance process v(t) (Heston-style): dv = kappa(1-v)dt + eta*sqrt(v)dW_v,
correlated to the rate Brownian(s) via rho. sigma_eff(t) = sigma(t)*sqrt(v(t)).
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .base import RateModel, CalibratedRateModel
from ._shared import (
    mean_reversion_shift, bootstrap_sigma, PiecewiseSigma, fit_theta,
    fit_skew_smile, year_frac, shift_date, simulate_cir_variance_step,
)
from ..calibration import priors as _priors


@dataclass
class _CalibratedLGM(CalibratedRateModel):
    curve: object
    mean_reversion: List[float]       # [a] or [a1, a2]
    sigma: List[PiecewiseSigma]       # one per factor
    rho: np.ndarray                   # n_factors x n_factors correlation of state Brownians
    sv: Optional[_priors.SVPriors] = None   # None for non-SV variants
    assumptions: dict = field(default_factory=dict)

    @property
    def n_factors(self):
        return len(self.mean_reversion)

    def zeta(self, factor_index: int, t: float) -> float:
        return self.sigma[factor_index].zeta(t)

    def discount_factor(self, state, t: float, T: float, T_date=None, t_date=None) -> float:
        """T_date/t_date: the ACTUAL target dates, when the caller already
        has them (e.g. build_market_states_at / StateImpliedCurve, which
        know the real trade cashflow date being priced) -- bypasses
        shift_date()'s date->year_frac->date round-trip, which loses up to
        0.5 days of precision and was producing ~$1 noise on ~$300k+
        notionals in end-to-end pricing tests. Falls back to shift_date()
        when only the year-fraction is available (internal math, tests)."""
        df0_T = self.curve.discount(T_date if T_date is not None else
                                     (self.curve.ref_date if T == 0 else shift_date(self.curve.ref_date, T)))
        df0_t = self.curve.discount(t_date if t_date is not None else
                                     (self.curve.ref_date if t == 0 else shift_date(self.curve.ref_date, t)))
        u = T - t
        H = [mean_reversion_shift(a_i, u) for a_i in self.mean_reversion]
        zeta = [self.sigma[i].zeta(t) for i in range(self.n_factors)]
        linear = sum(-H[i] * state[i] for i in range(self.n_factors))
        # 0.5 * H^T Sigma(t) H, Sigma(t)_ij = rho_ij * sqrt(zeta_i * zeta_j):
        # the own-variance (i=j) terms reduce to 0.5*H_i^2*zeta_i as in the 1F
        # case; the i!=j cross terms are the correlated-factor covariance the
        # 1F formula has no room for -- omitting them would make the DF
        # expectation miss the input curve whenever rho_12 != 0.
        quad = 0.0
        for i in range(self.n_factors):
            for j in range(self.n_factors):
                cov_ij = self.rho[i, j] * math.sqrt(max(zeta[i], 0.0) * max(zeta[j], 0.0))
                quad += H[i] * H[j] * cov_ij
        return (df0_T / df0_t) * math.exp(linear - 0.5 * quad)

    def discount_factor_batch(self, state_batch: np.ndarray, t: float, T: float) -> np.ndarray:
        """Vectorized discount_factor: state_batch shape (n_paths, n_factors)
        -> (n_paths,) array of DF(t,T), one per path. Same formula as
        discount_factor(), just batched over paths with numpy instead of a
        per-path Python call -- used by build_market_states_at, which
        otherwise calls discount_factor/level_at once per path per
        regression date and dominates pricing wall-clock at scale."""
        df0_T = self.curve.discount(self.curve.ref_date if T == 0 else shift_date(self.curve.ref_date, T))
        df0_t = self.curve.discount(self.curve.ref_date if t == 0 else shift_date(self.curve.ref_date, t))
        u = T - t
        H = np.array([mean_reversion_shift(a_i, u) for a_i in self.mean_reversion])
        zeta = np.array([self.sigma[i].zeta(t) for i in range(self.n_factors)])
        linear = -state_batch @ H   # (n_paths,)
        quad = 0.0
        for i in range(self.n_factors):
            for j in range(self.n_factors):
                cov_ij = self.rho[i, j] * math.sqrt(max(zeta[i], 0.0) * max(zeta[j], 0.0))
                quad += H[i] * H[j] * cov_ij
        return (df0_T / df0_t) * np.exp(linear - 0.5 * quad)

    def short_rate_integral(self, state, t: float) -> float:
        """integral_0^t r(s) ds implied by the state, i.e. -ln(DF(0,t)) -- the
        exact risk-neutral numeraire log, used by GBM/FXGBM to drift equity/FX
        off this SAME simulated rate path rather than a static curve."""
        return -math.log(self.discount_factor(state, 0.0, t))

    def simulate_paths(self, n_paths: int, horizon_dates, rng, external_z=None):
        times = [year_frac(self.curve.ref_date, d) for d in horizon_dates]
        times = [0.0] + times
        n_steps = len(times) - 1
        nf = self.n_factors
        has_sv = self.sv is not None

        x = np.zeros((n_paths, n_steps + 1, nf))
        # v(t) is a dimensionless CIR variance MULTIPLIER with stationary mean
        # 1 (dv = kappa(1-v)dt + eta*sqrt(v)dW), so eff_sigma = sigma(t)*sqrt(v(t))
        # has stationary level sigma(t) -- the calibrated vol level already lives
        # in sigma(t), not in v. Initializing v to self.sv.theta here would
        # double-apply the vol scale (theta is in vol^2 units, not a
        # dimensionless multiplier) and silently mute the simulated vol to
        # near zero -- this was the source of a real forward-matching bias
        # caught by test_lgm_forward_matching.py's SV cases.
        v = np.ones((n_paths, n_steps + 1, nf))

        chol = np.linalg.cholesky(self.rho) if nf > 1 else np.array([[1.0]])

        for step in range(n_steps):
            t0, t1 = times[step], times[step + 1]
            if t1 <= t0:
                # No time elapsed -- e.g. horizon_dates' first entry IS
                # ref_date itself (times[0]=0.0, times[1]=0.0 too), which the
                # 3-stage pipeline's SimulationGrid always includes. Copy
                # state forward with zero diffusion rather than stepping
                # over a spurious dt floor -- using max(t1-t0, 1e-10) here
                # previously injected a tiny but real eff_sigma*sqrt(1e-10)*z
                # noise term into what should be an EXACT zero state at
                # t=0, which on a large-notional trade (e.g. $100M) showed
                # up as real dollar noise (~$1-4/path) in NPV0 at ref_date.
                x[:, step + 1, :] = x[:, step, :]
                v[:, step + 1, :] = v[:, step, :]
                continue
            dt = t1 - t0
            if external_z is not None:
                z = external_z[:, step, :]
            else:
                z = rng.standard_normal((n_paths, nf)) @ chol.T
            for i, a_i in enumerate(self.mean_reversion):
                sigma_t = self.sigma[i].at(t0)
                if has_sv:
                    vz = rng.standard_normal(n_paths)
                    vprev = v[:, step, i]
                    v[:, step + 1, i] = simulate_cir_variance_step(vprev, self.sv.kappa, self.sv.eta, dt, vz)
                    eff_sigma = sigma_t * np.sqrt(np.maximum(v[:, step, i], 0.0))
                else:
                    eff_sigma = sigma_t
                # LGM state is driftless Brownian (all curve-dependence is in H(t)
                # via discount_factor(), not in the state SDE) -- this driftless
                # dynamic is exactly what makes forward matching exact.
                x[:, step + 1, i] = x[:, step, i] + eff_sigma * np.sqrt(dt) * z[:, i]

        return x[:, 1:, :]  # drop t=0 slice; shape (n_paths, len(horizon_dates), n_factors)


class LGM1F(RateModel):
    name = "LGM1F"

    def __init__(self, mean_reversion_a: float = 0.03):
        self.a = mean_reversion_a

    def calibrate(self, curve, vol_surface, **kw) -> _CalibratedLGM:
        pillars, sigmas = bootstrap_sigma(vol_surface.atm_term_structure())
        return _CalibratedLGM(
            curve=curve,
            mean_reversion=[self.a],
            sigma=[PiecewiseSigma(pillars, sigmas)],
            rho=np.array([[1.0]]),
            assumptions={"mean_reversion_a": self.a, "sigma_source": "ATM term vol, bootstrapped"},
        )


class LGM1FSV(RateModel):
    name = "LGM1F_SV"

    def __init__(self, mean_reversion_a: float = 0.03, sv_prior: _priors.SVPriors = None):
        self.a = mean_reversion_a
        self.sv_prior = sv_prior

    def calibrate(self, curve, vol_surface, **kw) -> _CalibratedLGM:
        atm = vol_surface.atm_term_structure()
        pillars, sigmas = bootstrap_sigma(atm)
        theta = fit_theta(vol_surface)
        prior = self.sv_prior or _priors.rate_sv_prior(theta)
        rho_smile, eta_smile = fit_skew_smile(vol_surface)
        sv = _priors.SVPriors(
            kappa=prior.kappa,
            theta=theta,
            eta=eta_smile if eta_smile is not None else prior.eta,
            rho=rho_smile if rho_smile is not None else prior.rho,
        )
        return _CalibratedLGM(
            curve=curve,
            mean_reversion=[self.a],
            sigma=[PiecewiseSigma(pillars, sigmas)],
            rho=np.array([[1.0]]),
            sv=sv,
            assumptions={
                "mean_reversion_a": self.a,
                "sigma_source": "ATM term vol, bootstrapped",
                "theta_source": "ATM term vol level (identifiable)",
                "eta_source": "smile curvature" if eta_smile is not None else "literature prior (surface under-determined)",
                "rho_source": "smile skew" if rho_smile is not None else "literature prior (surface under-determined)",
                "kappa_source": "literature prior (no swaption/cap grid to identify mean-reversion of variance)",
            },
        )


class LGM2F(RateModel):
    name = "LGM2F"

    def __init__(self, mean_reversion_a1: float = 0.03, mean_reversion_a2: float = 0.30, rho_12: float = -0.7):
        self.a1, self.a2, self.rho_12 = mean_reversion_a1, mean_reversion_a2, rho_12

    def calibrate(self, curve, vol_surface, **kw) -> _CalibratedLGM:
        atm = vol_surface.atm_term_structure()
        pillars, sigmas = bootstrap_sigma(atm)
        # split total ATM vol between the short (a1) and long (a2) factors;
        # standard 2F LGM convention: factor 2 carries a fixed fraction of
        # total variance (documented, literature-typical: 30%)
        frac2 = 0.30
        sigmas1 = [s * math.sqrt(1 - frac2) for s in sigmas]
        sigmas2 = [s * math.sqrt(frac2) for s in sigmas]
        rho = np.array([[1.0, self.rho_12], [self.rho_12, 1.0]])
        return _CalibratedLGM(
            curve=curve,
            mean_reversion=[self.a1, self.a2],
            sigma=[PiecewiseSigma(pillars, sigmas1), PiecewiseSigma(pillars, sigmas2)],
            rho=rho,
            assumptions={
                "mean_reversion_a1": self.a1, "mean_reversion_a2": self.a2, "rho_12": self.rho_12,
                "variance_split": f"{1-frac2:.0%}/{frac2:.0%} short/long factor (literature-typical prior)",
            },
        )


class LGM2FSV(RateModel):
    name = "LGM2F_SV"

    def __init__(self, mean_reversion_a1: float = 0.03, mean_reversion_a2: float = 0.30,
                 rho_12: float = -0.7, sv_prior: _priors.SVPriors = None):
        self.a1, self.a2, self.rho_12 = mean_reversion_a1, mean_reversion_a2, rho_12
        self.sv_prior = sv_prior

    def calibrate(self, curve, vol_surface, **kw) -> _CalibratedLGM:
        atm = vol_surface.atm_term_structure()
        pillars, sigmas = bootstrap_sigma(atm)
        frac2 = 0.30
        sigmas1 = [s * math.sqrt(1 - frac2) for s in sigmas]
        sigmas2 = [s * math.sqrt(frac2) for s in sigmas]
        rho = np.array([[1.0, self.rho_12], [self.rho_12, 1.0]])
        theta = fit_theta(vol_surface)
        prior = self.sv_prior or _priors.rate_sv_prior(theta)
        rho_smile, eta_smile = fit_skew_smile(vol_surface)
        sv = _priors.SVPriors(
            kappa=prior.kappa, theta=theta,
            eta=eta_smile if eta_smile is not None else prior.eta,
            rho=rho_smile if rho_smile is not None else prior.rho,
        )
        return _CalibratedLGM(
            curve=curve,
            mean_reversion=[self.a1, self.a2],
            sigma=[PiecewiseSigma(pillars, sigmas1), PiecewiseSigma(pillars, sigmas2)],
            rho=rho,
            sv=sv,
            assumptions={
                "mean_reversion_a1": self.a1, "mean_reversion_a2": self.a2, "rho_12": self.rho_12,
                "variance_split": f"{1-frac2:.0%}/{frac2:.0%} short/long factor (literature-typical prior)",
                "theta_source": "ATM term vol level (identifiable)",
                "eta_source": "smile curvature" if eta_smile is not None else "literature prior (surface under-determined)",
                "rho_source": "smile skew" if rho_smile is not None else "literature prior (surface under-determined)",
                "kappa_source": "literature prior (no swaption/cap grid to identify mean-reversion of variance)",
            },
        )
