"""
Exposure metrics from a CurveResult's NPV0/NPV10 curves, per Slide 8/9's
Margin Period of Risk (MPoR) convention -- FORWARD-looking from each anchor:

    Default is assumed to happen AT anchor t. VM on a given date is the NPV
    of the trade on the PRIOR business day (Slide 9: "collected/posted
    variation margin ... is the NPV of the trade on the prior day on the
    path"), so the last valid mark before default is NPV0(t - 1bd) (the
    "value AT that date" curve). VM then discontinues (the counterparty
    defaulted) while it takes mpor_days business days to actually close out
    the position -- during that window the market keeps moving, so the
    exposure that materializes is NPV10(t) (the MPoR curve's value AT t,
    which by CurveResult's own definition is priced at t + mpor_days):

    exposure(path, t) = collateralized_exposure(npv10, t) - collateralized_exposure(npv0, t-1bd)

Metrics, at BOTH 95% and 99% confidence:
    EE(t)         = mean( max(exposure(:, t), 0) )
    PFE_95/99(t)  = 95th / 99th percentile( exposure(:, t) )
    MPE_95/99     = max_t PFE_95/99(t)
    EEPE_95/99    = time-weighted average of EE(t) over [0, 1yr] (standard Basel
                    EEPE definition -- averaged over the first year or the life
                    of the netting set if shorter)
    tail-EE_95/99(t) = mean(exposure(:,t) | exposure(:,t) > PFE_95/99(t))  --
                    conditional mean above the quantile threshold
    tail-EEPE_95/99  = time-weighted average of tail-EE_95/99(t) over [0,1yr]

(EE/EEPE are means, not quantiles, so "EE at 95%" isn't a standard risk
metric on its own -- the tail-EE/tail-EEPE variants above are what "both
confidence levels on every metric" concretely means: the conditional mean
ABOVE each confidence level's threshold, confirmed design.)

Two edge cases in the window differencing, both driven by trades entering or
leaving the book mid-window:

1. The very first anchor(s) have no valid t-1bd VM mark yet (it would
   predate ref_date) -- dropped from the reported curve.
2. A trade that MATURES anywhere within [t-1bd, t+mpor_days bd] drops from
   live NPV to 0 at settlement -- excluded from BOTH curves' sums for that
   window so settlement doesn't masquerade as exposure.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .netting.margin import MarginModel
from .simulation.grid import add_business_days


@dataclass
class ExposureProfile:
    owner_id: str
    dates: List
    ee: List[float]
    nee: List[float]
    pfe_95: List[float]
    pfe_99: List[float]
    tail_ee_95: List[float]
    tail_ee_99: List[float]
    mpe_95: float
    mpe_99: float
    eepe: float
    tail_eepe_95: float
    tail_eepe_99: float


def _exposure_all_paths(owner, curve_result, curve, margin_model, date, n_paths, exclude_trade_ids):
    return np.array([
        owner.collateralized_exposure(curve_result, curve, margin_model, p, date, exclude_trade_ids)
        for p in range(n_paths)
    ])


def _trades_maturing_in_window(curve_result, window_start, window_end) -> tuple:
    """Trade ids whose maturity falls within [window_start, window_end]
    (inclusive both ends: price_curves' own maturity guard keeps a trade
    live THROUGH its own maturity date, so the NPV->0 transition can land
    right at a boundary) -- excluded from both curves so settlement doesn't
    masquerade as a market-risk move."""
    return tuple(tid for tid, maturity in curve_result.trade_maturities.items() if window_start <= maturity <= window_end)


def _time_weighted_average(dates, values, ref_date, horizon_years=1.0):
    """Basel EEPE: time-weighted average of EE(t) over [0, horizon_years],
    trapezoidal between consecutive anchors, restricted to dates within the
    averaging window."""
    pts = [((d - ref_date).days / 365.0, v) for d, v in zip(dates, values) if (d - ref_date).days / 365.0 <= horizon_years]
    if len(pts) < 2:
        return pts[0][1] if pts else 0.0
    total, weight = 0.0, 0.0
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        total += 0.5 * (v0 + v1) * dt
        weight += dt
    return total / weight if weight > 0 else pts[-1][1]


def compute_exposure_profile(counterparty, curve_result, anchor_dates, ref_date,
                              margin_model: MarginModel = None, mpor_days: int = 10,
                              vm_lag_days: int = 1) -> ExposureProfile:
    if margin_model is None:
        from .netting.margin import ZeroMargin
        margin_model = ZeroMargin()

    n_paths = curve_result.n_paths
    dates, ee, nee, pfe_95, pfe_99, tail_ee_95, tail_ee_99 = [], [], [], [], [], [], []
    for t in anchor_dates:
        vm_date = add_business_days(t, -vm_lag_days)
        closeout_date = t   # NPV10's own definition already prices at t + mpor_days internally
        if vm_date < ref_date:
            continue

        exclude = _trades_maturing_in_window(curve_result, vm_date, add_business_days(t, mpor_days))
        exp_vm = _exposure_all_paths(counterparty, curve_result, "npv0", margin_model, vm_date, n_paths, exclude)
        exp_closeout = _exposure_all_paths(counterparty, curve_result, "npv10", margin_model, closeout_date, n_paths, exclude)

        exposure = exp_closeout - exp_vm
        dates.append(t)
        ee.append(float(np.mean(np.maximum(exposure, 0.0))))
        # NEE (negative expected exposure): the DVA-feeding mirror of EE --
        # mean of the NEGATIVE part of exposure, i.e. what WE would owe if
        # WE default. Always <= 0 (mean of a min-with-zero array).
        nee.append(float(np.mean(np.minimum(exposure, 0.0))))
        p95 = float(np.quantile(exposure, 0.95))
        p99 = float(np.quantile(exposure, 0.99))
        pfe_95.append(p95)
        pfe_99.append(p99)
        above_95 = exposure[exposure > p95]
        above_99 = exposure[exposure > p99]
        tail_ee_95.append(float(np.mean(above_95)) if above_95.size else p95)
        tail_ee_99.append(float(np.mean(above_99)) if above_99.size else p99)

    eepe = _time_weighted_average(dates, ee, ref_date)
    tail_eepe_95 = _time_weighted_average(dates, tail_ee_95, ref_date)
    tail_eepe_99 = _time_weighted_average(dates, tail_ee_99, ref_date)

    return ExposureProfile(
        owner_id=counterparty.id, dates=dates, ee=ee, nee=nee, pfe_95=pfe_95, pfe_99=pfe_99,
        tail_ee_95=tail_ee_95, tail_ee_99=tail_ee_99,
        mpe_95=max(pfe_95) if pfe_95 else 0.0, mpe_99=max(pfe_99) if pfe_99 else 0.0,
        eepe=eepe, tail_eepe_95=tail_eepe_95, tail_eepe_99=tail_eepe_99,
    )


class _BookTotal:
    """Sum of collateralized exposure across ALL counterparties -- explicitly
    NOT netted across them (no netting agreement spans counterparties), just
    a book-wide total for context alongside the per-counterparty curves."""

    id = "BOOK_TOTAL"

    def __init__(self, counterparties):
        self._counterparties = counterparties

    def collateralized_exposure(self, curve_result, curve, margin_model, path, date, exclude_trade_ids=()):
        return sum(c.collateralized_exposure(curve_result, curve, margin_model, path, date, exclude_trade_ids)
                   for c in self._counterparties)


def compute_all_profiles(counterparties, curve_result, anchor_dates, ref_date,
                          margin_model: MarginModel = None) -> Dict[str, ExposureProfile]:
    """One ExposureProfile per counterparty, plus 'BOOK_TOTAL'."""
    profiles = {}
    for c in counterparties:
        profiles[c.id] = compute_exposure_profile(c, curve_result, anchor_dates, ref_date, margin_model)
    profiles["BOOK_TOTAL"] = compute_exposure_profile(
        _BookTotal(counterparties), curve_result, anchor_dates, ref_date, margin_model)
    return profiles


def compute_per_trade_exposure_profile(trade_id: str, curve_result_for_trade, anchor_dates, ref_date,
                                        margin_model: MarginModel = None) -> ExposureProfile:
    """One trade's own ExposureProfile, from ITS OWN independent CurveResult
    (risk_engine.pricing.price_curves_per_trade output) -- mechanically just
    compute_exposure_profile against a single-trade NettingSet/Counterparty,
    which needs no new netting logic: a 1-trade NettingSet.netted_npv sums
    over exactly one trade id, i.e. degenerates to that trade's own NPV, so
    no cross-trade summation (and therefore no cross-trade correlation
    assumption) is involved here at all."""
    from .netting.netting_set import NettingSet
    from .netting.counterparty import Counterparty
    solo = Counterparty(id=trade_id, netting_sets=[
        NettingSet(id=trade_id, counterparty_id=trade_id, trade_ids=[trade_id])
    ])
    return compute_exposure_profile(solo, curve_result_for_trade, anchor_dates, ref_date, margin_model)


def compute_per_trade_profiles(trades: Dict[str, object], curve_results_by_trade_id: Dict[str, object],
                                anchor_dates, ref_date, margin_model: MarginModel = None) -> Dict[str, ExposureProfile]:
    """{trade_id: ExposureProfile} -- the per-trade analogue of
    compute_all_profiles, for risk_engine.simulation.per_trade's independent
    simulation mode. Does NOT include a 'BOOK_TOTAL'/counterparty-level
    entry -- see risk_engine.netting.independent_aggregate for the
    (explicitly labeled, NOT a netted/correlated number) counterparty-level
    combination appropriate for independently-simulated trades."""
    return {
        tid: compute_per_trade_exposure_profile(tid, curve_results_by_trade_id[tid], anchor_dates, ref_date, margin_model)
        for tid in trades
    }
