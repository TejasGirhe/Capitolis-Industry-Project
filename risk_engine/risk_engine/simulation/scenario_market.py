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
                            equity_dividend_rates: dict, reporting_ccy: str = "USD") -> List[MarketState]:
    """One MarketState per path, at target_date -- interpolated from the
    fixed-grid cached path via each model's state_at(). target_date need not
    be on the simulation grid.

    joint_result: risk_engine.simulation.joint.JointSimResult, simulated on
        the dates whose year-fractions are sim_times (same order).
    equity_dividend_rates: {isin: q} static market data, passed through
        unchanged -- dividend rate sets the simulation drift, it is not
        itself simulated (README Sec.3/MARKET_DATA.md Sec.3).
    """
    t = year_fraction_of(joint_result.ref_date, target_date)

    rate_state_by_factor: Dict[object, np.ndarray] = {}
    discount_curve_by_ccy = {}
    for factor, model in joint_result.rate_models.items():
        cached_path = joint_result.rate_states[factor]
        state_at_t = model.state_at(cached_path, sim_times, t)   # (n_paths, n_state_vars)
        rate_state_by_factor[factor] = state_at_t
        if isinstance(factor, RateFactor):
            discount_curve_by_ccy[factor.currency] = (model, state_at_t)

    spot_level_by_factor: Dict[object, np.ndarray] = {}
    for factor, model in joint_result.spot_models.items():
        cached_path = joint_result.spot_states[factor]
        state_at_t = model.state_at(cached_path, sim_times, t)   # (n_paths, n_state_vars)
        drift_factor = _drift_rate_factor_for(joint_result, factor)
        rate_state_at_t = rate_state_by_factor.get(drift_factor)
        if hasattr(model, "level_at_batch") and rate_state_at_t is not None:
            # vectorized path: build_market_states_at is called once per
            # regression date per curve, so a per-path Python loop here
            # (37 equity factors x ~100 dates x n_paths x 2 curves) dominates
            # pricing wall-clock at scale -- level_at_batch does the same
            # math with numpy across all paths at once.
            levels = model.level_at_batch(state_at_t, t, rate_state_at_t)
        else:
            n_paths = state_at_t.shape[0]
            levels = np.array([
                model.level_at(state_at_t[p], t, rate_state=rate_state_at_t[p] if rate_state_at_t is not None else None)
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
