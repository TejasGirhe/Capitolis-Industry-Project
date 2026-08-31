"""
Vega: bump-and-reprice sensitivity on the exposure engine already built
(price_curves -> compute_all_profiles). NOT a Basel III / SA-CCR concept --
SA-CCR's PFE multiplier uses a fixed supervisory volatility factor per
asset class (see sa_ccr.SUPERVISORY_FACTOR), not a shocked recompute, so
there is nothing in the regulatory framework this "vega" feeds into. This
is a risk-desk sensitivity metric only: how much do EE/PFE/MPE/EEPE move
for a +1-vol-point shock to one factor group's volatility surface.

Cost-scoped as a FORWARD difference (shocked case only, one extra full
pipeline run per factor group), not central (shocked-up minus shocked-down,
two extra runs per factor group) -- vega = (shocked_metric - base_metric) /
bump_size, computed by the caller against an ALREADY-COMPUTED base case, so
this module never re-runs the base case itself.
"""
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from ..netting import build_netting_hierarchy
from ..exposure import compute_all_profiles
from ..pricing import price_curves


@dataclass
class ScenarioInputs:
    """Everything needed to build ONE joint simulation + pricing run, with
    hooks for a caller to inject a vol bump on exactly one factor group
    before calibration -- built once for the base case, then reused with a
    bump applied for each shocked run so curve/spot/correlation inputs stay
    identical across base and shocked (only the targeted vol surface moves).
    """
    trades: Dict[str, object]
    equity_dividend_rates: Dict[str, float]
    ref_date: object
    grid_dates: List           # simulation.grid.build_simulation_grid(...).dates -- precache dates
    anchor_dates: List          # simulation.grid.reporting_anchors(...) -- what compute_all_profiles reports at
    regression_dates: List       # simulation.grid.collect_regression_dates(...) -- what price_curves prices at
                                  # (superset of anchor_dates: also includes trade cashflow dates + MPoR
                                  # window endpoints; compute_all_profiles needs THIS set to exist inside
                                  # the CurveResult it's given, not just the bare anchors, or its internal
                                  # t-mpor_days lookups KeyError)
    build_joint_simulator: Callable[[float], object]
    """build_joint_simulator(vol_bump: float) -> JointSimulator, fully wired
    (add_rate/add_spot already called) with `vol_bump` added to EVERY vol
    surface belonging to the target factor group and 0.0 elsewhere -- the
    caller (greeks_report.py) owns exactly which factor group is bumped;
    this module is agnostic to that, it only calls with bump_size then 0.0."""


EE_MAX = "EE_max"
MPE_95 = "MPE_95"
MPE_99 = "MPE_99"
EEPE = "EEPE"
METRIC_KEYS = (EE_MAX, MPE_95, MPE_99, EEPE)


def _summarize(profiles) -> Dict[str, Dict[str, float]]:
    """{owner_id: {metric_key: value}} from a compute_all_profiles() result."""
    out = {}
    for owner_id, p in profiles.items():
        out[owner_id] = {
            EE_MAX: max(p.ee) if p.ee else 0.0,
            MPE_95: p.mpe_95,
            MPE_99: p.mpe_99,
            EEPE: p.eepe,
        }
    return out


def run_scenario(scenario: ScenarioInputs, vol_bump: float, n_paths: int, n_workers=None,
                  rng_seed: int = 42, mpor_days: int = 10) -> Dict[str, Dict[str, float]]:
    """One full precache -> price_curves -> compute_all_profiles run, with
    `vol_bump` applied via scenario.build_joint_simulator. Returns the
    per-counterparty (+ BOOK_TOTAL) metric summary -- NOT the raw
    ExposureProfile objects, since vega only needs scalar deltas per
    metric, not full curves, per the confirmed scoping."""
    sim = scenario.build_joint_simulator(vol_bump)
    rng = np.random.default_rng(rng_seed)
    precache = sim.simulate(_market_state_stub(scenario.ref_date), n_paths=n_paths,
                             horizon_dates=scenario.grid_dates, rng=rng, ref_date=scenario.ref_date)
    result = price_curves(scenario.trades, precache, scenario.regression_dates,
                           equity_dividend_rates=scenario.equity_dividend_rates,
                           mpor_days=mpor_days, n_workers=n_workers)
    counterparties = build_netting_hierarchy(scenario.trades)
    profiles = compute_all_profiles(counterparties, result, scenario.anchor_dates, scenario.ref_date)
    return _summarize(profiles)


def _market_state_stub(ref_date):
    from capitolis_pricers.market import MarketState
    return MarketState(ref_date=ref_date)


def bump_vol_and_reprice(scenario: ScenarioInputs, factor_group: str, bump_size: float,
                          base_metrics: Dict[str, Dict[str, float]], n_paths: int,
                          n_workers=None) -> Dict[str, Dict[str, float]]:
    """Runs ONE shocked scenario (vol_bump=bump_size) and returns
    {owner_id: {metric_key: vega}} = (shocked - base) / bump_size for every
    owner/metric already present in base_metrics.

    factor_group is bookkeeping only here (used in the returned dict's
    caller-facing label, e.g. in greeks_report.py) -- the actual bump
    targeting happens inside scenario.build_joint_simulator, which the
    caller constructs per factor group (rate/equity/fx) before calling
    this function.
    """
    t0 = time.time()
    shocked_metrics = run_scenario(scenario, bump_size, n_paths, n_workers)
    elapsed = time.time() - t0

    vega = {}
    for owner_id, base in base_metrics.items():
        shocked = shocked_metrics.get(owner_id, {})
        vega[owner_id] = {
            metric: (shocked.get(metric, 0.0) - base.get(metric, 0.0)) / bump_size
            for metric in METRIC_KEYS
        }
    return {"factor_group": factor_group, "bump_size": bump_size, "elapsed_seconds": elapsed, "vega": vega}
