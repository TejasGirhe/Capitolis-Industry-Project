"""
Correlation-FREE counterparty aggregate, for combining per-trade exposure
profiles that came from INDEPENDENT simulations (risk_engine.simulation.
per_trade) -- explicitly NOT a call into NettingSet.netted_npv /
Counterparty.collateralized_exposure, which assume every trade in a netting
set was priced against the SAME joint scenario per path index. Summing
independently-simulated trades' NPVs at a shared path index would silently
misrepresent unrelated scenarios as a correlated netting number -- worse
than not netting, since it would look identical to a real one.

Every value this module returns is a DIFFERENT KIND OF NUMBER than
risk_engine.exposure.ExposureProfile's counterparty-level output, and must
never be compared to or substituted for it:

    EE_aggregate(t)  = sum_i EE_i(t)
        Valid exactly: E[sum_i X_i] = sum_i E[X_i] regardless of whether the
        X_i are correlated or not -- linearity of expectation. This is the
        one metric here that is NOT an approximation.

    PFE_aggregate_qq(t) = sum_i PFE_i_qq(t)
        NOT a true joint percentile. Summing independent trades' OWN
        marginal quantiles is a CONSERVATIVE UPPER BOUND on the quantile of
        their sum whenever they are independent (comonotonic sums achieve
        the maximum possible quantile of a sum for given marginals; true
        independence gives a strictly SMALLER quantile than the sum of
        marginals in general, by a diversification argument) -- so this
        overstates risk relative to what a genuine joint simulation would
        show, in the opposite direction from ignoring correlation entirely.
        Documented explicitly so it is never read as "the" PFE.

    MPE_aggregate_qq = max_t PFE_aggregate_qq(t)
"""
from dataclasses import dataclass
from typing import Dict, List

from ..exposure import ExposureProfile


@dataclass
class IndependentAggregateProfile:
    """NOT an ExposureProfile -- deliberately a different type so it can
    never be silently passed to code (e.g. plotting.plot_exposure_profiles)
    expecting a real, correlation-aware exposure number without an
    explicit, visible conversion step."""
    counterparty_id: str
    trade_ids: List[str]
    dates: List
    ee_aggregate: List[float]
    pfe_aggregate_95: List[float]
    pfe_aggregate_99: List[float]
    mpe_aggregate_95: float
    mpe_aggregate_99: float

    def label(self) -> str:
        return (f"{self.counterparty_id} (INDEPENDENT-TRADE AGGREGATE -- "
                f"sum of {len(self.trade_ids)} trades' own marginal EE/PFE, "
                f"NOT a netted/correlated exposure number; PFE is a conservative "
                f"upper bound, not a true joint percentile)")


def aggregate_independent_profiles(counterparty_id: str, trade_profiles: Dict[str, ExposureProfile]) -> IndependentAggregateProfile:
    """trade_profiles: {trade_id: ExposureProfile} for every trade belonging
    to `counterparty_id` (from exposure.compute_per_trade_profiles, filtered
    to one counterparty's trades by the caller -- this function doesn't know
    about counterparty membership itself, matching build_netting_hierarchy's
    grouping being the caller's job).

    All profiles must share the same `dates` (they will, if computed from
    the same anchor_dates/ref_date as usual) -- this function does not
    attempt to align mismatched date grids.
    """
    trade_ids = sorted(trade_profiles.keys())
    if not trade_ids:
        return IndependentAggregateProfile(counterparty_id, [], [], [], [], [], 0.0, 0.0)

    dates = trade_profiles[trade_ids[0]].dates
    for tid in trade_ids:
        if trade_profiles[tid].dates != dates:
            raise ValueError(
                f"trade '{tid}' has a different date grid than '{trade_ids[0]}' -- "
                f"aggregate_independent_profiles requires all trades share one anchor_dates set"
            )

    n = len(dates)
    ee_agg = [sum(trade_profiles[tid].ee[i] for tid in trade_ids) for i in range(n)]
    pfe95_agg = [sum(trade_profiles[tid].pfe_95[i] for tid in trade_ids) for i in range(n)]
    pfe99_agg = [sum(trade_profiles[tid].pfe_99[i] for tid in trade_ids) for i in range(n)]

    return IndependentAggregateProfile(
        counterparty_id=counterparty_id, trade_ids=trade_ids, dates=dates,
        ee_aggregate=ee_agg, pfe_aggregate_95=pfe95_agg, pfe_aggregate_99=pfe99_agg,
        mpe_aggregate_95=max(pfe95_agg) if pfe95_agg else 0.0,
        mpe_aggregate_99=max(pfe99_agg) if pfe99_agg else 0.0,
    )


def aggregate_all_independent_profiles(trades: Dict[str, object], trade_profiles: Dict[str, ExposureProfile]) -> Dict[str, IndependentAggregateProfile]:
    """{counterparty_id: IndependentAggregateProfile} -- groups trade_profiles
    by each trade's .counterparty attribute (same grouping
    netting.build_netting_hierarchy uses) and aggregates within each group."""
    by_cpty: Dict[str, Dict[str, ExposureProfile]] = {}
    for tid, trade in trades.items():
        cpty = trade.counterparty
        by_cpty.setdefault(cpty, {})[tid] = trade_profiles[tid]

    return {cpty: aggregate_independent_profiles(cpty, profiles) for cpty, profiles in by_cpty.items()}
