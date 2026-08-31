"""
CVA / DVA / FVA report -- xVA (Slide 5/10's optional extra-credit item),
built on the already-validated exposure engine (EE/NEE profiles) and real
sourced credit-tier data.

Credit spreads: no free source publishes name-specific CDS curves for
CPTY_A/B/C (they aren't real entities) or Capitolis itself -- confirmed by
direct investigation (same conclusion as the equity/rate vol-surface
search). What IS real and free: FRED's Moody's AAA/BAA seasoned corporate
bond yields minus the 10Y Treasury give a genuine RATING-TIER spread proxy.
Counterparties are tiered by book size (largest book -> AAA-tier, best
assumed credit; smallest -> BAA-tier) -- see risk_engine/market_data/credit.py
for the full sourcing chain and its documented limitations.

Market data driving the underlying exposure simulation (USD curve, equity
spot/vol/correlation, FX spot/vol) is ALSO real, sourced via
_sourced_book.build_sourced_book() -- see that module for exactly which
field is real vs. a documented placeholder.

    python risk_engine/examples/xva_report.py [n_paths] [n_workers] [ref_date]
"""
import json
import os
import sys
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
JSON_OUT_PATH = os.path.join(OUT_DIR, "xva_report.json")


def main(n_paths=10_000, n_workers=None, pricing_date=None):
    book = build_sourced_book(ref_date=pricing_date)
    ref_date, trades = book["ref_date"], book["trades"]
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]
    usd_curve = book["market_data"]["usd_curve"]

    print("\nSourcing credit-tier spreads (FRED Moody's AAA/BAA corporate yields)...")
    aaa = fetch_rating_tier_spread("AAA")
    baa = fetch_rating_tier_spread("BAA")
    print(f"  AAA tier spread (over 10Y UST): {aaa:.4%}")
    print(f"  BAA tier spread (over 10Y UST): {baa:.4%}")
    credit_curves = build_credit_curves(trades, ref_date)
    own_credit = build_own_credit_curve(ref_date)
    spreads = counterparty_spreads(trades)
    for cpty, cc in sorted(credit_curves.items()):
        print(f"  {cpty}: spread={spreads[cpty]:.4%} (recovery={cc.recovery:.0%})")

    sim = book["build_simulator"]()

    print(f"\nPrecaching {n_paths} paths x {len(grid.dates)} simulation dates "
          f"({len(book['market_data']['correlations'])} sourced pairwise correlations)...")
    import time
    t0 = time.time()
    rng = np.random.default_rng(42)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                             rng=rng, ref_date=ref_date)
    print(f"  precache: {time.time()-t0:.1f}s")

    print(f"Pricing {len(trades)} trades x {n_paths} paths x {len(regression_dates)} regression dates "
          f"(n_workers={n_workers or os.cpu_count()})...")
    t0 = time.time()
    result = price_curves(trades, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=n_workers)
    print(f"  price_curves: {time.time()-t0:.1f}s")

    counterparties = build_netting_hierarchy(trades)
    profiles = compute_all_profiles(counterparties, result, anchors, ref_date)
    per_cpty_profiles = {k: v for k, v in profiles.items() if k != "BOOK_TOTAL"}

    # Funding spread: reuses the SAME BAA-tier proxy as own_credit (a mid-tier,
    # unrated-but-active market participant) -- a placeholder a real desk
    # would replace with its own actual funding curve.
    funding_spread = baa
    xva = compute_all_xva(per_cpty_profiles, credit_curves, own_credit, funding_spread, usd_curve, ref_date)

    print("\n=== xVA Report (CVA/DVA/FVA -- CDS spreads are a RATING-TIER PROXY, not real counterparty curves) ===")
    total_cva = total_dva = total_fva = 0.0
    for cpty in sorted(xva):
        res = xva[cpty]
        print(f"  {cpty:12s} CVA={res.cva:12,.2f}  DVA={res.dva:12,.2f}  FVA={res.fva:12,.2f}  Net={res.net_xva:12,.2f}")
        total_cva += res.cva
        total_dva += res.dva
        total_fva += res.fva
    print(f"  {'BOOK_TOTAL':12s} CVA={total_cva:12,.2f}  DVA={total_dva:12,.2f}  FVA={total_fva:12,.2f}  "
          f"Net={total_cva - total_dva + total_fva:12,.2f}")

    json_payload = {
        "ref_date": ref_date.isoformat(),
        "n_paths": n_paths,
        "data_source": "real (FRED + Yahoo Finance) exposure simulation + real FRED rating-tier credit proxy",
        "credit_tier_spreads": {"AAA": aaa, "BAA": baa},
        "counterparty_spreads": spreads,
        "funding_spread": funding_spread,
        "xva": {
            c: {"cva": r.cva, "dva": r.dva, "fva": r.fva, "net_xva": r.net_xva}
            for c, r in xva.items()
        },
        "book_total": {"cva": total_cva, "dva": total_dva, "fva": total_fva,
                       "net_xva": total_cva - total_dva + total_fva},
    }
    with open(JSON_OUT_PATH, "w") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"\nSaved data to {JSON_OUT_PATH}")
    return xva


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pricing_date = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
    main(n_paths, n_workers, pricing_date)
