"""
Full end-to-end rebuild-and-report run: real market data -> joint Monte
Carlo simulation -> trade-level AND netting-set-level exposure profiles ->
CVA/DVA/FVA at both levels -> SA-CCR RC/PFE-addon/EAD chain -- everything
this run needs to populate a regulator-facing status report, from ONE
consistent joint simulation (same paths, same correlation structure feed
both the trade-level and netting-set-level numbers, so they are mutually
consistent rather than drawn from separate simulations).

Trade-level and netting-set-level exposure both read off the SAME
CurveResult (see pricing.CurveResult's module docstring): a per-trade
"netting set" of exactly 1 trade correctly degenerates
NettingSet.netted_npv to that trade's own NPV, so no separate independent-
per-trade simulation is needed or used here (deliberately -- this keeps
trade-level and book-level numbers on the same footing, unlike
risk_engine.simulation.per_trade's independent-simulation mode, which is
for a different purpose: isolating one trade's own convergence behavior).

    python risk_engine/examples/full_regulatory_run.py [n_paths] [n_workers] [seed]
"""
import json
import os
import platform
import sys
import time
from datetime import date

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.pricing import price_curves
from risk_engine.netting import build_netting_hierarchy
from risk_engine.exposure import compute_all_profiles, compute_per_trade_exposure_profile
from risk_engine.market_data.credit import (
    build_credit_curves, build_own_credit_curve, fetch_rating_tier_spread, counterparty_spreads,
)
from risk_engine.xva import compute_xva_report
from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.examples.report_inputs import compute_report_inputs

OUT_DIR = os.path.dirname(__file__)
JSON_OUT_PATH = os.path.join(OUT_DIR, "benchmark_results", "full_regulatory_run_results.json")


def main(n_paths=10_000, n_workers=None, seed=42, pricing_date=None):
    run_start = time.time()
    print(f"{'='*78}\nFULL REGULATORY RUN -- clean-slate rebuild\n{'='*78}")
    print(f"n_paths={n_paths:,}  seed={seed}  n_workers={n_workers or os.cpu_count()}  "
          f"machine={platform.platform()}")

    # ---------------------------------------------------------- 1. DATA INGESTION
    print("\n[1/6] Sourcing market data (Bloomberg-preferred, FRED/Yahoo/Databento fallback)...")
    t0 = time.time()
    book = build_sourced_book(ref_date=pricing_date)
    sourcing_s = time.time() - t0
    ref_date, trades = book["ref_date"], book["trades"]
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]
    usd_curve = book["market_data"]["usd_curve"]
    print(f"  sourcing: {sourcing_s:.1f}s  |  {len(trades)} trades  |  "
          f"{len(book['factors'].equities)} equity factors  |  ref_date={ref_date.isoformat()}")

    # ---------------------------------------------------------- 2. CREDIT CURVES
    print("\n[2/6] Sourcing credit-tier spreads (FRED Moody's AAA/BAA corporate yields)...")
    aaa = fetch_rating_tier_spread("AAA")
    baa = fetch_rating_tier_spread("BAA")
    credit_curves = build_credit_curves(trades, ref_date)
    own_credit = build_own_credit_curve(ref_date)
    spreads = counterparty_spreads(trades)
    funding_spread = baa
    for cpty, cc in sorted(credit_curves.items()):
        print(f"  {cpty}: spread={spreads[cpty]:.4%}  recovery={cc.recovery:.0%}")

    # ---------------------------------------------------------- 3. JOINT SIMULATION
    sim = book["build_simulator"]()
    print(f"\n[3/6] Simulating {n_paths:,} paths x {len(grid.dates)} simulation-grid dates "
          f"(seed={seed})...")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths,
                            horizon_dates=grid.dates, rng=rng, ref_date=ref_date)
    sim_s = time.time() - t0
    print(f"  simulate(): {sim_s:.1f}s")

    # ---------------------------------------------------------- 4. PRICING
    print(f"\n[4/6] Pricing {len(trades)} trades x {n_paths:,} paths x "
          f"{len(regression_dates)} regression dates (n_workers={n_workers or os.cpu_count()})...")
    t0 = time.time()
    result = price_curves(trades, precache, regression_dates, equity_dividend_rates=equity_div,
                          n_workers=n_workers)
    price_s = time.time() - t0
    print(f"  price_curves(): {price_s:.1f}s")

    # ---------------------------------------------------------- 5. EXPOSURE + xVA -- NETTING-SET LEVEL
    print("\n[5/6] Netting-set-level exposure profiles + xVA...")
    counterparties = build_netting_hierarchy(trades)
    profiles = compute_all_profiles(counterparties, result, anchors, ref_date)
    per_cpty_profiles = {k: v for k, v in profiles.items() if k != "BOOK_TOTAL"}

    cpty_xva = {}
    for cpty, profile in per_cpty_profiles.items():
        if cpty not in credit_curves:
            continue
        cpty_xva[cpty] = compute_xva_report(profile, credit_curves[cpty], own_credit,
                                            funding_spread, usd_curve, ref_date)

    # gross (pre-netting) exposure: sum of each COUNTERPARTY's own trades'
    # UN-netted EE, i.e. sum of trade-level EE within that counterparty,
    # vs. the netting-set's own (netted) EE -- the netting benefit is the
    # difference, reported explicitly below.
    trade_to_cpty = {tid: t.counterparty for tid, t in trades.items()}

    # ---------------------------------------------------------- 6. EXPOSURE + xVA -- TRADE LEVEL
    print("[6/6] Trade-level exposure profiles + xVA (same joint paths, same CurveResult)...")
    trade_profiles = {}
    trade_xva = {}
    for tid in trades:
        prof = compute_per_trade_exposure_profile(tid, result, anchors, ref_date)
        trade_profiles[tid] = prof
        cpty = trade_to_cpty[tid]
        if cpty in credit_curves:
            trade_xva[tid] = compute_xva_report(prof, credit_curves[cpty], own_credit,
                                                funding_spread, usd_curve, ref_date)

    # gross exposure per counterparty = sum of that counterparty's trades' own EE curves
    # (summed pointwise per anchor date, then take the max for a gross MPE-equivalent) --
    # a CONSERVATIVE, non-netted reference point, not a claim that this is itself a
    # correct joint quantile (see netting.independent_aggregate's docstring for why
    # summing marginals overstates a joint PFE quantile in general; EE sums exactly
    # by linearity of expectation regardless, so the gross-vs-net EE/EEPE comparison
    # below is exact, while a gross-vs-net PFE comparison would not be).
    from risk_engine.exposure import _time_weighted_average
    gross_ee_by_cpty = {}
    for cpty in per_cpty_profiles:
        member_tids = [tid for tid, c in trade_to_cpty.items() if c == cpty]
        if not member_tids:
            continue
        # dates may differ trade-to-trade near each trade's own maturity edge
        # (compute_per_trade_exposure_profile drops any anchor before that
        # trade's own vm_date >= ref_date check) -- use the counterparty's
        # OWN netted profile's date grid as the common axis, which is a
        # strict superset/aligned reference already computed above, and
        # sum only the trades that have an entry at each date.
        dates = per_cpty_profiles[cpty].dates
        date_index = {tid: {d: i for i, d in enumerate(trade_profiles[tid].dates)} for tid in member_tids}
        gross_ee = []
        for d in dates:
            gross_ee.append(sum(
                trade_profiles[tid].ee[date_index[tid][d]] for tid in member_tids if d in date_index[tid]
            ))
        gross_eepe = _time_weighted_average(dates, gross_ee, ref_date)
        gross_ee_by_cpty[cpty] = {"dates": [d.isoformat() for d in dates], "gross_ee": gross_ee,
                                  "gross_max_ee": max(gross_ee) if gross_ee else 0.0,
                                  "gross_eepe": gross_eepe}

    # ---------------------------------------------------------- SA-CCR (report_inputs.py's validated CRE52 chain)
    print("\nComputing SA-CCR RC/PFE-addon/EAD chain (CRE52, uncollateralised assumption)...")
    saccr_inputs = compute_report_inputs(book, restrict_to_trade_ids=list(trades.keys()))

    run_elapsed = time.time() - run_start

    # ================================================================ REPORT PRINTOUT
    print(f"\n{'='*78}\nRUN COMPLETE -- {run_elapsed:.1f}s total\n{'='*78}")
    print(f"Reporting date: {ref_date.isoformat()}   Currency: USD (reporting_ccy)")
    print(f"Paths: {n_paths:,}   Seed: {seed}   Simulation grid: {len(grid.dates)} dates "
          f"({grid.dates[0].isoformat()} to {grid.dates[-1].isoformat()})")
    print(f"Regression dates: {len(regression_dates)}   Reporting anchors: {len(anchors)}")

    print(f"\n--- TRADE LEVEL ---")
    print(f"{'trade_id':<14s}{'cpty':<10s}{'MtM(t0)':>16s}{'EE_max':>14s}{'EEPE':>14s}"
          f"{'PFE95_max':>14s}{'PFE99_max':>14s}{'CVA':>12s}{'DVA':>12s}{'FVA':>12s}{'NetXVA':>12s}")
    for tid in sorted(trades):
        prof = trade_profiles[tid]
        mtm0 = saccr_inputs["trades"][tid]["mtm0"]
        xr = trade_xva.get(tid)
        cva, dva, fva, net = (xr.cva, xr.dva, xr.fva, xr.net_xva) if xr else (0, 0, 0, 0)
        print(f"{tid:<14s}{trade_to_cpty[tid]:<10s}{mtm0:>16,.0f}{max(prof.ee) if prof.ee else 0:>14,.0f}"
              f"{prof.eepe:>14,.0f}{prof.mpe_95:>14,.0f}{prof.mpe_99:>14,.0f}"
              f"{cva:>12,.0f}{dva:>12,.0f}{fva:>12,.0f}{net:>12,.0f}")

    print(f"\n--- NETTING-SET / COUNTERPARTY LEVEL ---")
    print(f"{'counterparty':<14s}{'RC':>12s}{'PFE(addon)':>14s}{'EAD':>14s}"
          f"{'Net_EE_max':>14s}{'Gross_EE_max':>14s}{'Netting_benefit':>18s}"
          f"{'CVA':>12s}{'DVA':>12s}{'FVA':>12s}{'NetXVA':>12s}")
    total = {"cva": 0.0, "dva": 0.0, "fva": 0.0}
    for cpty in sorted(per_cpty_profiles):
        prof = per_cpty_profiles[cpty]
        sc = saccr_inputs["saccr"].get(cpty, {})
        xr = cpty_xva.get(cpty)
        cva, dva, fva, net = (xr.cva, xr.dva, xr.fva, xr.net_xva) if xr else (0, 0, 0, 0)
        net_ee_max = max(prof.ee) if prof.ee else 0.0
        gross_ee_max = gross_ee_by_cpty.get(cpty, {}).get("gross_max_ee", 0.0)
        benefit = gross_ee_max - net_ee_max
        print(f"{cpty:<14s}{sc.get('RC', 0):>12,.0f}{sc.get('PFE', 0):>14,.0f}{sc.get('EAD', 0):>14,.0f}"
              f"{net_ee_max:>14,.0f}{gross_ee_max:>14,.0f}{benefit:>18,.0f}"
              f"{cva:>12,.0f}{dva:>12,.0f}{fva:>12,.0f}{net:>12,.0f}")
        total["cva"] += cva
        total["dva"] += dva
        total["fva"] += fva
    print(f"{'BOOK_TOTAL':<14s}{'':>12s}{'':>14s}{'':>14s}{'':>14s}{'':>14s}{'':>18s}"
          f"{total['cva']:>12,.0f}{total['dva']:>12,.0f}{total['fva']:>12,.0f}"
          f"{total['cva']-total['dva']+total['fva']:>12,.0f}")

    # ================================================================ SAVE
    def _profile_dict(p):
        return {"owner_id": p.owner_id, "dates": [d.isoformat() for d in p.dates], "ee": p.ee, "nee": p.nee,
                "pfe_95": p.pfe_95, "pfe_99": p.pfe_99, "mpe_95": p.mpe_95, "mpe_99": p.mpe_99,
                "eepe": p.eepe, "tail_ee_95": p.tail_ee_95, "tail_ee_99": p.tail_ee_99,
                "tail_eepe_95": p.tail_eepe_95, "tail_eepe_99": p.tail_eepe_99}

    def _xva_dict(x):
        return {"cva": x.cva, "dva": x.dva, "fva": x.fva, "net_xva": x.net_xva} if x else None

    payload = {
        "run_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ref_date": ref_date.isoformat(),
            "n_paths": n_paths, "seed": seed, "n_workers": n_workers or os.cpu_count(),
            "n_sim_grid_dates": len(grid.dates),
            "sim_grid_start": grid.dates[0].isoformat(), "sim_grid_end": grid.dates[-1].isoformat(),
            "n_regression_dates": len(regression_dates), "n_reporting_anchors": len(anchors),
            "reporting_ccy": "USD",
            "sourcing_s": sourcing_s, "simulate_s": sim_s, "price_curves_s": price_s,
            "total_run_s": run_elapsed,
            "machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "python_version": platform.python_version(), "cpu_count": os.cpu_count()},
        },
        "credit": {"aaa_tier_spread": aaa, "baa_tier_spread": baa, "counterparty_spreads": spreads,
                  "funding_spread": funding_spread},
        "trade_level": {
            tid: {"counterparty": trade_to_cpty[tid], "mtm0": saccr_inputs["trades"][tid]["mtm0"],
                 "asset_class": saccr_inputs["trades"][tid]["asset_class"],
                 "profile": _profile_dict(trade_profiles[tid]), "xva": _xva_dict(trade_xva.get(tid))}
            for tid in trades
        },
        "netting_set_level": {
            cpty: {"profile": _profile_dict(per_cpty_profiles[cpty]),
                  "gross_ee": gross_ee_by_cpty.get(cpty, {}),
                  "saccr": saccr_inputs["saccr"].get(cpty, {}),
                  "xva": _xva_dict(cpty_xva.get(cpty))}
            for cpty in per_cpty_profiles
        },
        "book_total_xva": {**total, "net_xva": total["cva"] - total["dva"] + total["fva"]},
        "saccr_assumptions": saccr_inputs["assumptions"],
    }
    os.makedirs(os.path.dirname(JSON_OUT_PATH), exist_ok=True)
    with open(JSON_OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved full results to {JSON_OUT_PATH}")
    return payload


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    main(n_paths, n_workers, seed)
