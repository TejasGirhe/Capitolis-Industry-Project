"""
Two independent grids, cleanly separated per the confirmed 3-stage design:

- build_simulation_grid: the FIXED, small grid factors are actually Monte
  Carlo simulated on (the "precache"). Independent of any trade's cashflow
  dates or the MPoR window -- sized purely for good interpolation fidelity
  (dense enough that any date pricing needs is never far from a real
  simulated node).
- collect_regression_dates: every date pricing actually needs a value AT --
  a trade's own cashflow/reset dates, plus each reporting anchor's MPoR
  window endpoints (t and t+10bd, with t-1bd as the VM mark). These are
  "regression dates" in the sense of "dates you regress/interpolate a value
  for from the cached simulation," not Longstaff-Schwartz continuation-value
  regression (none of these 3 products have early-exercise optionality that
  would need that). Every regression date gets its factor state via
  risk_engine.simulation.interpolate / models/*.state_at() -- none of them
  need to be on the simulation grid.

No holiday calendar (capitolis_pricers.daycount is deliberately calendar-
agnostic too -- see its module docstring); "business day" here means
Mon-Fri only, matching that same simplification.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List

from capitolis_pricers.daycount import add_months, schedule_forward, to_date


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5


def add_business_days(d: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = d
    while remaining > 0:
        cur += timedelta(days=step)
        if _is_business_day(cur):
            remaining -= 1
    return cur


@dataclass
class SimulationGrid:
    ref_date: date
    horizon: date
    dates: List[date]   # the ONLY dates JointSimulator.simulate() runs on


def build_simulation_grid(ref_date, max_maturity_date) -> SimulationGrid:
    """Fixed, MPoR-independent simulation grid: weekly through month 3
    (where reset/cashflow dates cluster densest in this book), monthly
    through year 1, quarterly beyond, out to max_maturity_date + 1yr (no
    trade has exposure past its own maturity, so anything further is
    necessarily flat/zero -- same horizon logic as before)."""
    ref_date = to_date(ref_date)
    max_maturity_date = to_date(max_maturity_date)
    horizon = add_months(max_maturity_date, 12)

    weekly = []
    cur = ref_date + timedelta(days=7)
    month3_end = ref_date + timedelta(days=90)
    while cur <= month3_end:
        weekly.append(cur)
        cur += timedelta(days=7)

    monthly = []
    k = 4
    while True:
        d = add_months(ref_date, k)
        if d > horizon or (d - ref_date).days > 365:
            break
        monthly.append(d)
        k += 1

    quarterly = []
    k = 1
    while True:
        d = add_months(ref_date, 12 + 3 * k)
        if d > horizon:
            break
        quarterly.append(d)
        k += 1
    if not quarterly or quarterly[-1] < horizon:
        quarterly.append(horizon)

    dates = sorted(set([ref_date] + weekly + monthly + quarterly))
    return SimulationGrid(ref_date=ref_date, horizon=horizon, dates=dates)


def reporting_anchors(ref_date, max_maturity_date) -> List[date]:
    """Monthly through year 1, quarterly beyond -- the dates EE/PFE/etc. get
    reported at. Kept separate from the simulation grid's own density (which
    is now independent of reporting cadence)."""
    ref_date = to_date(ref_date)
    max_maturity_date = to_date(max_maturity_date)
    horizon = add_months(max_maturity_date, 12)

    monthly = []
    k = 1
    while True:
        d = add_months(ref_date, k)
        if d > horizon or (d - ref_date).days > 365:
            break
        monthly.append(d)
        k += 1

    quarterly = []
    k = 1
    while True:
        d = add_months(ref_date, 12 + 3 * k)
        if d > horizon:
            break
        quarterly.append(d)
        k += 1
    if not quarterly or quarterly[-1] < horizon:
        quarterly.append(horizon)

    return sorted(set(monthly + quarterly))


def trade_cashflow_dates(trade) -> List[date]:
    """Every cashflow/reset/maturity date structurally defined by one trade
    -- used both for regression-date collection and for cashflow graphs."""
    if hasattr(trade, "forward_date"):     # BondForwardTrade
        return [trade.forward_date]
    if hasattr(trade, "start_date") and hasattr(trade, "end_date"):  # EquityTRS, BondTRS
        return schedule_forward(trade.start_date, trade.end_date, trade.reset_m)
    raise AttributeError(f"{type(trade).__name__} has no recognized cashflow-date attributes")


def collect_regression_dates(trades: Dict[str, object], anchors: List[date], ref_date=None,
                              mpor_days: int = 10, vm_lag_days: int = 1) -> List[date]:
    """Every date pricing needs a value AT: each trade's own cashflow dates,
    plus each reporting anchor's MPoR window endpoints (t-vm_lag_days,
    t, t+mpor_days), plus ref_date itself (the natural "today" pricing point
    -- always worth having a value at, e.g. for a deterministic-pricing
    sanity check). None of these need to be on the simulation grid --
    state_at() interpolates them from the cached simulation path."""
    dates = set()
    if ref_date is not None:
        dates.add(to_date(ref_date))
    for trade in trades.values():
        dates.update(trade_cashflow_dates(trade))
    for t in anchors:
        dates.add(t)
        dates.add(add_business_days(t, -vm_lag_days))
        dates.add(add_business_days(t, mpor_days))
    return sorted(dates)
