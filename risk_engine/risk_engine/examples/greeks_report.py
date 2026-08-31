"""
Greeks: SA-CCR delta (Basel III Annex 4, closed-form, full book, cheap) +
bump-and-reprice vega (this session's own sensitivity metric, one extra
full exposure-pipeline run per factor group: rate, equity, FX).

SA-CCR delta feeds Slide 15's EAD = alpha * (RC + PFE_addon) framing --
it is NOT the same thing as vega, and vega is NOT a Basel III concept (SA-CCR
uses a fixed supervisory vol factor per asset class, not a shocked
recompute). See risk_engine/greeks/sa_ccr.py and vega.py docstrings.

Market data (base case, before any vega bump) is REAL, sourced from FRED +
Yahoo Finance via _sourced_book.build_sourced_book() -- see that module for
exactly which field is real vs. a documented placeholder (rate vol surface
has no free source; the vega BUMP is applied on top of whatever base vol
level was sourced/assumed for that factor group).

    python risk_engine/examples/greeks_report.py [n_paths] [n_workers] [ref_date]

n_paths here is the VEGA path count. A reduced path count (500-2000) was
tried first for speed, but the 95th/99th-percentile PFE/MPE tail metrics
were unstable there -- confirmed NOT a bump-size artifact (still broken at
a 1bp shock): a 99th percentile needs roughly 50-100+ tail observations to
be stable, and 500 paths gives only ~5. Default n_paths is therefore
10,000, matching exposure_profile.py's scale -- this script runs the full
pipeline 4x (1 base + 3 bumped), so budget ~4x exposure_profile.py's
per-run cost for the full report; run in the background.
"""
import json
import os
import sys
from datetime import date

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.greeks import aggregate_delta, GreeksReport
from risk_engine.greeks.vega import ScenarioInputs, run_scenario, bump_vol_and_reprice
from risk_engine.examples._sourced_book import build_sourced_book


OUT_DIR = os.path.dirname(__file__)
JSON_OUT_PATH = os.path.join(OUT_DIR, "greeks_report.json")

# Uniform 1bp vol shock across every factor group.
VEGA_BUMP = {"rate": 0.0001, "equity": 0.0001, "fx": 0.0001}


def main(n_paths=10_000, n_workers=None, pricing_date=None):
    book = build_sourced_book(ref_date=pricing_date)
    ref_date, trades = book["ref_date"], book["trades"]
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    print("\nComputing SA-CCR delta (Basel III Annex 4)...")
    sa_ccr = aggregate_delta(trades, ref_date)
    for cpty in sorted(sa_ccr):
        print(f"  {cpty}: {sa_ccr[cpty]}")

    def build_joint_simulator(factor_group: str):
        """vol_bump -> JointSimulator, applying the bump to ONLY
        `factor_group`'s sourced vol surface, base sourced vol elsewhere --
        reuses book['build_simulator'], which already knows how to add a
        bump to one factor group's VolSurface on top of the sourced level."""
        def _build(vol_bump: float):
            return book["build_simulator"]({factor_group: vol_bump})
        return _build

    base_scenario = ScenarioInputs(
        trades=trades, equity_dividend_rates=equity_div, ref_date=ref_date,
        grid_dates=grid.dates, anchor_dates=anchors, regression_dates=regression_dates,
        build_joint_simulator=build_joint_simulator("rate"),  # factor_group irrelevant at bump=0.0
    )
    print(f"\nRunning base case ({n_paths} paths)...")
    base_metrics = run_scenario(base_scenario, vol_bump=0.0, n_paths=n_paths, n_workers=n_workers)
    for owner_id, m in sorted(base_metrics.items()):
        print(f"  {owner_id:12s} {m}")

    vega_runs = []
    for factor_group in ("rate", "equity", "fx"):
        bump_size = VEGA_BUMP[factor_group]
        print(f"\nRunning {factor_group} vol +{bump_size:.2%} bump ({n_paths} paths)...")
        scenario = ScenarioInputs(
            trades=trades, equity_dividend_rates=equity_div, ref_date=ref_date,
            grid_dates=grid.dates, anchor_dates=anchors, regression_dates=regression_dates,
            build_joint_simulator=build_joint_simulator(factor_group),
        )
        run = bump_vol_and_reprice(scenario, factor_group, bump_size, base_metrics, n_paths, n_workers)
        vega_runs.append(run)
        print(f"  ({run['elapsed_seconds']:.1f}s)")
        for cpty in sorted(run["vega"]):
            print(f"  {cpty:12s} {run['vega'][cpty]}")

    report = GreeksReport(ref_date=ref_date, sa_ccr_delta=sa_ccr, vega_runs=vega_runs)
    print("\n" + report.summary())

    payload = report.to_dict()
    payload["data_source"] = "real (FRED + Yahoo Finance) base case -- see _sourced_book.py"
    with open(JSON_OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved data to {JSON_OUT_PATH}")
    return report


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pricing_date = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
    main(n_paths, n_workers, pricing_date)
