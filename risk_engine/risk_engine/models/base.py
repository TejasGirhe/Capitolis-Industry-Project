"""
Stochastic model interfaces -- split by factor kind for type safety.

Rate factors need a discount factor DF(t,T); spot factors (equity, FX) need a
level(t) (a price/rate, not a discount ratio). These are different contracts,
so RateModel/CalibratedRateModel and SpotModel/CalibratedSpotModel are
separate ABCs rather than one shared interface with NotImplementedError
defaults -- a rate model can never be handed to code expecting level_at(), and
vice versa; the mismatch is a missing-attribute/type error at the call site,
not a silent failure discovered deep inside a Monte Carlo loop.

Both families share the same two-stage shape:

    model = LGM1F()                                    # or GBM(), FXGBM()
    calibrated = model.calibrate(...)                   # forward + vol matching
    paths = calibrated.simulate_paths(n_paths, horizon_dates, rng)

`simulate_paths` accepts an optional `external_z`: pre-correlated standard
normal increments of shape (n_paths, n_steps, n_state_vars_for_this_factor),
supplied by risk_engine.simulation.joint.JointSimulator when this factor is
being simulated jointly with others under one N-factor correlation matrix.
When `external_z=None` (single-factor use), the model draws its own
independent increments from `rng` -- this is what the original single-factor
risk_engine.simulate() entry point still uses.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional


class RateModel(ABC):
    """A named, uncalibrated short-rate model (the LGM family)."""

    name: ClassVar[str]

    @abstractmethod
    def calibrate(self, curve, vol_surface, **priors) -> "CalibratedRateModel":
        """Fit model parameters to today's discount curve and vol surface.

        curve: capitolis_pricers.curves.Curve for the factor's currency.
        vol_surface: risk_engine.calibration.market_surface.VolSurface for this factor.
        priors: literature-typical overrides (see risk_engine.calibration.priors)
            for parameters the surface under-determines, e.g. kappa=, eta=, rho=.
        """
        raise NotImplementedError


class CalibratedRateModel(ABC):
    """A calibrated short-rate model: simulates state paths and reprices
    discount factors from a simulated state."""

    @abstractmethod
    def simulate_paths(self, n_paths: int, horizon_dates: List, rng, external_z=None):
        """Simulate n_paths state trajectories at the given horizon dates.
        Returns shape (n_paths, len(horizon_dates), n_state_vars)."""
        raise NotImplementedError

    @abstractmethod
    def discount_factor(self, state, t: float, T: float, T_date=None, t_date=None) -> float:
        """Analytic DF(t,T) given a simulated state vector at time t.

        This is the forward-matching guarantee made checkable: averaged over
        paths, discount_factor(state, t, T) must reproduce curve.discount(T)/
        curve.discount(t) used at calibration time, within Monte Carlo noise.

        T_date/t_date: optional actual target dates (bypasses a lossy
        date<->year-fraction round-trip when the caller already has the real
        date -- see models/lgm.py's _CalibratedLGM.discount_factor for why).
        """
        raise NotImplementedError

    def zeta(self, factor_index: int, t: float) -> float:
        """Accumulated variance for state dimension `factor_index` at time t
        (year-fraction from ref_date) -- the weighting function used by
        state_at()'s bridge interpolation. Overridden per concrete model
        (see models/lgm.py's _CalibratedLGM, which has one PiecewiseSigma
        per factor)."""
        raise NotImplementedError

    def state_at(self, cached_path, sim_times: List[float], t: float):
        """Interpolate this factor's state at year-fraction t from a path
        already simulated ONLY on sim_times (the fixed simulation grid) --
        no new Monte Carlo draw. Conditional-mean Brownian bridge, weighted
        by accumulated variance (zeta), not raw calendar time -- see
        risk_engine.simulation.interpolate for the derivation. Exact (no
        interpolation) when t lands exactly on a grid date.
        """
        from ..simulation.interpolate import interpolate_state
        n_factors = cached_path.shape[-1]
        cols = []
        for i in range(n_factors):
            cols.append(interpolate_state(cached_path[:, :, i:i + 1], sim_times, t,
                                           lambda tt, i=i: self.zeta(i, tt)))
        import numpy as np
        return np.concatenate(cols, axis=-1)


class SpotModel(ABC):
    """A named, uncalibrated spot-valued model (equity GBM, FX GBM)."""

    name: ClassVar[str]

    @abstractmethod
    def calibrate(self, spot: float, drift_source, vol_surface, **priors) -> "CalibratedSpotModel":
        """Fit model parameters to today's spot, a drift source, and a vol surface.

        spot: today's level (equity price in its own currency, or FX spot).
        drift_source: what supplies the risk-neutral drift r(t) (equity) or
            r_domestic(t) - r_foreign(t) (FX). Concretely a CalibratedRateModel
            (for equity: the paired USD rate model, so drift follows the
            *actual simulated* rate path per scenario, not a static curve) or
            an (rate_model, implied_foreign_curve) pair (for FX -- see
            risk_engine.models.fx). This is a deliberate deviation from
            RateModel.calibrate(curve, ...): spot models need the calibrated
            rate model itself, not just a Curve, to stay path-consistent.
        vol_surface: risk_engine.calibration.market_surface.VolSurface for this factor.
        """
        raise NotImplementedError


class CalibratedSpotModel(ABC):
    """A calibrated spot-valued model: simulates state paths and reprices a
    level (not a discount ratio) from a simulated state."""

    @abstractmethod
    def simulate_paths(self, n_paths: int, horizon_dates: List, rng, external_z=None):
        """Simulate n_paths state trajectories at the given horizon dates.

        The returned state is diffusion-only (log-growth + auxiliary
        variance-process bookkeeping) -- it does NOT itself encode the
        risk-neutral drift, since that is path-dependent on the paired rate
        factor's own simulated path. level_at() combines this state with the
        paired rate_state (see its docstring) to produce the actual level.
        Returns shape (n_paths, len(horizon_dates), n_state_vars).
        """
        raise NotImplementedError

    @abstractmethod
    def level_at(self, state, t: float, rate_state=None) -> float:
        """The spot/FX level implied by a simulated state at time t.

        rate_state: the paired rate factor's own simulated state at t (from
            its simulate_paths() output), required when this model was
            calibrated against a CalibratedRateModel drift_source -- the
            level formula reads the risk-neutral drift off that state via
            the rate model's own discount_factor()/short_rate_integral(),
            not off a separately-stored copy. Models with a purely
            deterministic drift (none in this package yet) can ignore it.
        """
        raise NotImplementedError

    def zeta(self, t: float) -> float:
        """Accumulated variance at time t, the bridge-weighting function for
        state_at(). log_growth is a drift-corrected martingale, not a pure
        driftless Brownian motion (see models/gbm.py's module docstring), so
        zeta-weighted interpolation of it is exact for the deterministic-vol
        (non-SV) case and a documented approximation for -SV variants, where
        the effective vol depends on the stochastic CIR path -- confirmed
        tradeoff, favoring simplicity over exactly decomposing the drift and
        martingale components separately."""
        raise NotImplementedError

    def state_at(self, cached_path, sim_times: List[float], t: float):
        """Interpolate this factor's [log_growth, zeta_or_zero] state at
        year-fraction t from a path simulated ONLY on sim_times -- see
        CalibratedRateModel.state_at for the shared bridge-interpolation
        mechanics; the only difference here is the zeta() source."""
        from ..simulation.interpolate import interpolate_state
        import numpy as np
        n_factors = cached_path.shape[-1]
        cols = []
        for i in range(n_factors):
            cols.append(interpolate_state(cached_path[:, :, i:i + 1], sim_times, t, self.zeta))
        return np.concatenate(cols, axis=-1)
