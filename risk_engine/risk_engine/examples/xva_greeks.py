"""
xVA vega: CVA/DVA/FVA under a base case and under a +1bp vol shock to each
factor group (rate/equity/fx), reported as Base, Shocked, and Delta
(Shocked - Base) per counterparty -- the xVA analogue of greeks_report.py's
EE/MPE vega (risk_engine.greeks.vega), which tracks exposure metrics but not
CVA/DVA/FVA themselves.

CVA/DVA are direct linear functionals of a counterparty's EE/NEE profile
(risk_engine.xva.compute_cva/compute_dva), and EE/NEE already move under a
vol shock (that's exactly what greeks_report.py's vega already demonstrates)
-- so xVA vega is a cheap extension of exposure vega, not a new simulation
concept: run the SAME base/shocked scenarios exposure vega already runs,
just also feed the resulting ExposureProfiles into compute_all_xva instead
of only summarizing them into scalar EE/MPE metrics (which is all
risk_engine.greeks.vega.run_scenario keeps -- it discards the full
per-date ee/nee arrays xVA needs, hence this script re-runs rather than
reusing that module's saved output).

Credit curves (rating-tier proxy) and funding spread are IDENTICAL across
base and every shocked run -- only the exposure simulation's vol input
moves, exactly mirroring greeks_report.py's vega design (only the targeted
factor group's vol surface changes; everything else, including the credit
curves computed once up front here, stays fixed).

    python risk_engine/examples/xva_greeks.py [n_paths] [n_workers] [ref_date]
"""
import json
import os
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
from risk_engine.exposure import compute_all_profiles
from risk_engine.market_data.credit import (
    build_credit_curves, build_own_credit_curve, fetch_rating_tier_spread, counterparty_spreads,
)
from risk_engine.xva import compute_all_xva
from risk_engine.examples._sourced_book import build_sourced_book


OUT_DIR = os.path.dirname(__file__)
JSON_OUT_PATH = os.path.join(OUT_DIR, "xva_greeks.json")

VEGA_BUMP = {"rate": 0.0001, "equity": 0.0001, "fx": 0.0001}


def _run_xva(book, sim, credit_curves, own_credit, funding_spread, usd_curve, n_paths, n_workers, rng_seed=42):
    """One full precache -> price_curves -> profiles -> xVA run. Returns
    {counterparty: XVAResult} (per-counterparty only, no BOOK_TOTAL -- same
    scope as compute_all_xva, summed by the caller if a book total is
    wanted)."""
    ref_date, trades = book["ref_date"], book["trades"]
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    rng = np.random.default_rng(rng_seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                             rng=rng, ref_date=ref_date)
    result = price_curves(trades, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=n_workers)
    counterparties = build_netting_hierarchy(trades)
    profiles = compute_all_profiles(counterparties, result, anchors, ref_date)
    per_cpty_profiles = {k: v for k, v in profiles.items() if k != "BOOK_TOTAL"}
    return compute_all_xva(per_cpty_profiles, credit_curves, own_credit, funding_spread, usd_curve, ref_date)


def _totals(xva_by_cpty):
    cva = sum(r.cva for r in xva_by_cpty.values())
    dva = sum(r.dva for r in xva_by_cpty.values())
    fva = sum(r.fva for r in xva_by_cpty.values())
    return {"cva": cva, "dva": dva, "fva": fva, "net_xva": cva - dva + fva}


def main(n_paths=10_000, n_workers=None, pricing_date=None):
    book = build_sourced_book(ref_date=pricing_date)
    ref_date = book["ref_date"]
    usd_curve = book["market_data"]["usd_curve"]

    print("\nSourcing credit-tier spreads (FRED Moody's AAA/BAA corporate yields)...")
    aaa = fetch_rating_tier_spread("AAA")
    baa = fetch_rating_tier_spread("BAA")
    credit_curves = build_credit_curves(book["trades"], ref_date)
    own_credit = build_own_credit_curve(ref_date)
    spreads = counterparty_spreads(book["trades"])
    for cpty, cc in sorted(credit_curves.items()):
        print(f"  {cpty}: spread={spreads[cpty]:.4%} (recovery={cc.recovery:.0%})")
    funding_spread = baa   # same BAA-tier proxy xva_report.py uses

    print(f"\nRunning base case ({n_paths} paths)...")
    t0 = time.time()
    base_sim = book["build_simulator"]()
    base_xva = _run_xva(book, base_sim, credit_curves, own_credit, funding_spread, usd_curve, n_paths, n_workers)
    base_total = _totals(base_xva)
    print(f"  ({time.time()-t0:.1f}s)")
    for cpty in sorted(base_xva):
        r = base_xva[cpty]
        print(f"  {cpty:12s} CVA={r.cva:12,.2f}  DVA={r.dva:12,.2f}  FVA={r.fva:12,.2f}  Net={r.net_xva:12,.2f}")
    print(f"  {'BOOK_TOTAL':12s} CVA={base_total['cva']:12,.2f}  DVA={base_total['dva']:12,.2f}  "
          f"FVA={base_total['fva']:12,.2f}  Net={base_total['net_xva']:12,.2f}")

    shocked_runs = {}
    for factor_group, bump_size in VEGA_BUMP.items():
        print(f"\nRunning {factor_group} vol +{bump_size:.2%} bump ({n_paths} paths)...")
        t0 = time.time()
        shocked_sim = book["build_simulator"]({factor_group: bump_size})
        shocked_xva = _run_xva(book, shocked_sim, credit_curves, own_credit, funding_spread, usd_curve,
                                n_paths, n_workers)
        shocked_total = _totals(shocked_xva)
        print(f"  ({time.time()-t0:.1f}s)")
        shocked_runs[factor_group] = {"per_cpty": shocked_xva, "total": shocked_total}

    print("\n=== xVA Vega Report (Base / Shocked / Delta = Shocked - Base, per +1bp vol shock) ===")
    all_cptys = sorted(base_xva)
    for factor_group in VEGA_BUMP:
        print(f"\n--- {factor_group} vol +1bp ---")
        shocked = shocked_runs[factor_group]
        for cpty in all_cptys:
            b, s = base_xva[cpty], shocked["per_cpty"][cpty]
            print(f"  {cpty:12s} "
                  f"CVA  base={b.cva:11,.2f}  shocked={s.cva:11,.2f}  delta={s.cva-b.cva:11,.2f}")
            print(f"  {'':12s} "
                  f"DVA  base={b.dva:11,.2f}  shocked={s.dva:11,.2f}  delta={s.dva-b.dva:11,.2f}")
            print(f"  {'':12s} "
                  f"FVA  base={b.fva:11,.2f}  shocked={s.fva:11,.2f}  delta={s.fva-b.fva:11,.2f}")
            print(f"  {'':12s} "
                  f"Net  base={b.net_xva:11,.2f}  shocked={s.net_xva:11,.2f}  delta={s.net_xva-b.net_xva:11,.2f}")
        bt, st = base_total, shocked["total"]
        print(f"  {'BOOK_TOTAL':12s} "
              f"CVA  base={bt['cva']:11,.2f}  shocked={st['cva']:11,.2f}  delta={st['cva']-bt['cva']:11,.2f}")
        print(f"  {'':12s} "
              f"DVA  base={bt['dva']:11,.2f}  shocked={st['dva']:11,.2f}  delta={st['dva']-bt['dva']:11,.2f}")
        print(f"  {'':12s} "
              f"FVA  base={bt['fva']:11,.2f}  shocked={st['fva']:11,.2f}  delta={st['fva']-bt['fva']:11,.2f}")
        print(f"  {'':12s} "
              f"Net  base={bt['net_xva']:11,.2f}  shocked={st['net_xva']:11,.2f}  delta={st['net_xva']-bt['net_xva']:11,.2f}")

    def _xva_dict(r):
        return {"cva": r.cva, "dva": r.dva, "fva": r.fva, "net_xva": r.net_xva}

    payload = {
        "ref_date": ref_date.isoformat(),
        "n_paths": n_paths,
        "data_source": "real (FRED + Yahoo Finance) exposure simulation + real FRED rating-tier credit proxy",
        "credit_tier_spreads": {"AAA": aaa, "BAA": baa},
        "counterparty_spreads": spreads,
        "funding_spread": funding_spread,
        "base": {
            "per_cpty": {c: _xva_dict(r) for c, r in base_xva.items()},
            "book_total": base_total,
        },
        "shocks": {
            factor_group: {
                "bump_size": VEGA_BUMP[factor_group],
                "per_cpty": {c: _xva_dict(r) for c, r in shocked["per_cpty"].items()},
                "book_total": shocked["total"],
                "delta_per_cpty": {
                    c: {k: getattr(shocked["per_cpty"][c], k) - getattr(base_xva[c], k)
                        for k in ("cva", "dva", "fva")}
                    for c in all_cptys
                },
                "delta_book_total": {k: shocked["total"][k] - base_total[k] for k in ("cva", "dva", "fva", "net_xva")},
            }
            for factor_group, shocked in shocked_runs.items()
        },
    }
    with open(JSON_OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved data to {JSON_OUT_PATH}")
    return base_xva, shocked_runs


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pricing_date = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
    main(n_paths, n_workers, pricing_date)
