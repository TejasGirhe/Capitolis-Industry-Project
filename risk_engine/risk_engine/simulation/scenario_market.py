"""
Bridge: a JointSimResult (simulated only on the fixed simulation grid) ->
capitolis_pricers.market.MarketState objects at an ARBITRARY target date, one
per path, via each model's state_at() bridge interpolation -- so
pricer.npv(market) can be called on any regression date, not just simulated
grid dates. The pricers themselves are never touched.
"""
from typing import Dict, List

import numpy as np

from capitolis_pricers.market import MarketState
from capitolis_pricers.curves import FxCurve

from ..factors.types import RateFactor, EquityFactor, FxFactor
from ..models._shared import year_frac
from ._state_curve import StateImpliedCurve


def build_market_states_at(joint_result, sim_times: List[float], target_date,
                            equity_dividend_rates: dict, reporting_ccy: str = "USD",
                            add_bridge_noise: bool = True) -> List[MarketState]:
    """One MarketState per path, at target_date -- interpolated from the
    fixed-grid cached path via each model's state_at(). target_date need not
    be on the simulation grid.

    joint_result: risk_engine.simulation.joint.JointSimResult, simulated on
        the dates whose year-fractions are sim_times (same order).
    equity_dividend_rates: {isin: q} static market data, passed through
        unchanged -- dividend rate sets the simulation drift, it is not
        itself simulated (README Sec.3/MARKET_DATA.md Sec.3).
    add_bridge_noise: True (default -- changed from the original mean-only
        default; see risk_engine/examples/regulatory_readiness_report.html
        for why) adds the exact conditional-variance bridge noise at
        target_date, correlated across factors via
        joint_result.corr_chol -- the same Cholesky factor the grid-date
        draws used (risk_engine.simulation.joint.JointSimulator.simulate) --
        so an off-grid date's cross-factor correlation matches the main
        simulation's, not independent per-factor noise. Reproducible: the
        noise is a deterministic hash of (path_id, target_date, factor key),
        not a fresh RNG draw, so repeated calls (e.g. NPV0 vs NPV10, or
        separate multiprocess workers) see the same noise for the same
        (path, factor, date). See simulation.interpolate.draw_bridge_noise.
    """
    t = year_fraction_of(joint_result.ref_date, target_date)
    slot_ranges = joint_result.slot_ranges or {}
    corr_chol = joint_result.corr_chol if add_bridge_noise else None

    joint_bridge_z = None
    if corr_chol is not None:
        n_paths_bn = next(iter(joint_result.rate_states.values())).shape[0] if joint_result.rate_states \
            else next(iter(joint_result.spot_states.values())).shape[0]
        # ONE joint standard-normal draw across every driver of every factor,
        # correlated via the SAME Cholesky factor the grid-date draws used
        # (JointSimulator.simulate) -- this is what actually makes bridge
        # noise for the rate factor and, say, an equity factor CORRELATED
        # with each other at this date, not just correlated within a single
        # multi-driver factor's own internal dimensions. Drawing per-factor
        # sub-blocks separately (an earlier draft of this function) cannot
        # reproduce cross-factor correlation from a diagonal sub-block alone
        # -- the joint draw-then-slice pattern mirrors joint.py's own
        # z_indep @ chol.T exactly, just reproducibly seeded per (path, t)
        # instead of drawn from a live rng.
        from .interpolate import draw_bridge_noise
        n_drivers_total = corr_chol.shape[0]
        joint_bridge_z = draw_bridge_noise(np.arange(n_paths_bn), t, "joint_bridge",
                                            corr_chol=corr_chol, n_drivers=n_drivers_total)

    def _factor_z(factor, n_drivers):
        """This factor's own slice of the one joint correlated bridge-noise
        draw -- already correlated (within itself and to every other
        factor) via joint_bridge_z's construction above."""
        if joint_bridge_z is None or factor not in slot_ranges:
            return None
        s, n = slot_ranges[factor]
        assert n == n_drivers, f"driver count mismatch for {factor}: slot has {n}, model reports {n_drivers}"
        return joint_bridge_z[:, s:s + n]

    rate_state_by_factor: Dict[object, np.ndarray] = {}
    discount_curve_by_ccy = {}
    for factor, model in joint_result.rate_models.items():
        cached_path = joint_result.rate_states[factor]
        n_drivers = getattr(model, "n_factors", 1)
        state_at_t = model.state_at(cached_path, sim_times, t, add_bridge_noise=add_bridge_noise,
                                     seed_salt=str(factor), external_z=_factor_z(factor, n_drivers))
        rate_state_by_factor[factor] = state_at_t
        if isinstance(factor, RateFactor):
            discount_curve_by_ccy[factor.currency] = (model, state_at_t)

    drift_map = joint_result.drift_rate_factor_by_spot or {}
    foreign_map = joint_result.foreign_rate_factor_by_spot or {}

    spot_level_by_factor: Dict[object, np.ndarray] = {}
    for factor, model in joint_result.spot_models.items():
        cached_path = joint_result.spot_states[factor]
        state_at_t = model.state_at(cached_path, sim_times, t, add_bridge_noise=add_bridge_noise,
                                     seed_salt=str(factor), external_z=_factor_z(factor, 1))   # (n_paths, n_state_vars)
        drift_factor = drift_map.get(factor) or _drift_rate_factor_for(joint_result, factor)
        rate_state_at_t = rate_state_by_factor.get(drift_factor)
        foreign_factor = foreign_map.get(factor)
        foreign_rate_state_at_t = rate_state_by_factor.get(foreign_factor) if foreign_factor is not None else None
        if hasattr(model, "level_at_batch") and rate_state_at_t is not None:
            # vectorized path: build_market_states_at is called once per
            # regression date per curve, so a per-path Python loop here
            # (37 equity factors x ~100 dates x n_paths x 2 curves) dominates
            # pricing wall-clock at scale -- level_at_batch does the same
            # math with numpy across all paths at once.
            if foreign_rate_state_at_t is not None:
                levels = model.level_at_batch(state_at_t, t, rate_state_at_t,
                                              foreign_rate_state_batch=foreign_rate_state_at_t)
            else:
                levels = model.level_at_batch(state_at_t, t, rate_state_at_t)
        else:
            n_paths = state_at_t.shape[0]
            levels = np.array([
                model.level_at(state_at_t[p], t, rate_state=rate_state_at_t[p] if rate_state_at_t is not None else None,
                               **({"foreign_rate_state": foreign_rate_state_at_t[p]} if foreign_rate_state_at_t is not None else {}))
                for p in range(n_paths)
            ])
        spot_level_by_factor[factor] = levels

    n_paths = next(iter(rate_state_by_factor.values())).shape[0] if rate_state_by_factor else \
        next(iter(spot_level_by_factor.values())).shape[0]

    markets = []
    for p in range(n_paths):
        discount_curves = {
            ccy: StateImpliedCurve(model, state_at_t[p], t, target_date)
            for ccy, (model, state_at_t) in discount_curve_by_ccy.items()
        }
        equity_spots, fx_curves = {}, {}
        for factor, levels in spot_level_by_factor.items():
            level = float(levels[p])
            if isinstance(factor, EquityFactor):
                equity_spots[factor.isin] = level
            elif isinstance(factor, FxFactor):
                # pricers only ever call market.fx(...) (spot lookup, see
                # capitolis_pricers.market.MarketState.fx), never
                # FxCurve.forward() -- base_curve/quote_curve unset.
                fx_curves[(factor.base_ccy, factor.quote_ccy)] = FxCurve(factor.base_ccy, factor.quote_ccy, level)

        markets.append(MarketState(
            ref_date=target_date,
            reporting_ccy=reporting_ccy,
            discount_curves=discount_curves,
            equity_spots=equity_spots,
            equity_dividend_rates=equity_dividend_rates,
            fx_curves=fx_curves,
        ))
    return markets


def _drift_rate_factor_for(joint_result, spot_factor):
    """The rate factor a spot model's drift is path-consistent with.

    JointSimResult doesn't store this mapping directly (it lives on the
    JointSimulator that produced the result), so this falls back to "the
    book's single rate factor" when there's exactly one -- true for the
    current book (USD only) and the common case generally. Multi-currency
    books would need the mapping threaded through explicitly.
    """
    if len(joint_result.rate_models) == 1:
        return next(iter(joint_result.rate_models))
    raise ValueError(
        "Multiple rate factors present; build_market_states_at cannot infer which "
        "one drives this spot factor's drift without an explicit mapping. "
        "Pass drift_rate_factor_map explicitly (not yet supported by this helper)."
    )


def year_fraction_of(ref_date, d):
    return (d - ref_date).days / 365.0
