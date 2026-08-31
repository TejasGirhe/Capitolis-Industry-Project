"""
End-to-end EE / PFE_95 / PFE_99 / EEPE / MPE_95 / MPE_99 exposure profile for
the full 16-trade book, under joint LGM2F_SV (rates) + GBM_SV (equity) +
FXGBM_SV (FX) simulation, via the 3-stage pipeline:

    1. Precache: simulate every factor ONCE on a fixed, MPoR-independent grid.
    2. Pricing: interpolate factor states at every regression date (trade
       cashflow dates + each anchor's MPoR window endpoints) from the
       precache -- no re-simulation -- producing NPV0 (value at the date)
       and NPV10 (the MPoR curve, value 10 business days later).
    3. Aggregation: netting -> margin -> counterparty, computing EE/PFE_95/
       PFE_99/EEPE/tail-EE/MPE per Slide 8/9's forward-looking MPoR
       convention (ZeroMargin -- no collateral, per Slide 9's "assume no
       initial margin").

Market data (USD curve, equity spot/vol/correlation, FX spot/vol) is REAL,
sourced from FRED + Yahoo Finance via _sourced_book.build_sourced_book() --
see that module and risk_engine/market_data/build_market.py for exactly
which source backs each field and what stays a documented placeholder (rate
vol surface, dividend yields -- no free source exists for either).

Reference date defaults to TODAY (sourced data is "live" -- the valuation
date should match when it was observed), not a fixed backdated date. Several
book trades may already be mid-life or matured as of today; price_curves'
maturity guard handles this correctly (zeros a trade past its own maturity),
so a shorter-than-full-lifecycle exposure profile is expected, not a bug.

    python risk_engine/examples/exposure_profile.py [n_paths] [n_workers] [ref_date]
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
from risk_engine.plotting import plot_exposure_profiles
from risk_engine.visualization import plot_all_factor_fans, plot_all_cashflow_graphs
from risk_engine.examples._sourced_book import build_sourced_book


OUT_DIR = os.path.dirname(__file__)


def main(n_paths=10_000, n_workers=None, pricing_date=None, rate_model="LGM2F_SV"):
    suffix = "" if rate_model == "LGM2F_SV" else f"_{rate_model}"
    out_path_png = os.path.join(OUT_DIR, f"exposure_profile{suffix}.png")
    json_out_path = os.path.join(OUT_DIR, f"exposure_profile{suffix}.json")

    book = build_sourced_book(ref_date=pricing_date, rate_model=rate_model)
    ref_date, trades = book["ref_date"], book["trades"]
    factors, grid, anchors, regression_dates = book["factors"], book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    sim = book["build_simulator"]()
    rate_factor = factors.rates[0]

    print(f"\nPrecaching {n_paths} paths x {len(grid.dates)} simulation dates "
          f"({len(book['market_data']['correlations'])} sourced pairwise correlations)...")
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
    print(f"Counterparties: {[c.id for c in counterparties]}")

    t0 = time.time()
    profiles = compute_all_profiles(counterparties, result, anchors, ref_date)
    print(f"  exposure profiles: {time.time()-t0:.1f}s")

    for owner_id, p in sorted(profiles.items()):
        print(f"{owner_id:12s} MPE_99 = {p.mpe_99:15,.2f}  MPE_95 = {p.mpe_95:15,.2f}  "
              f"EEPE = {p.eepe:15,.2f}  max EE = {max(p.ee) if p.ee else 0.0:15,.2f}")

    out_path = plot_exposure_profiles(
        profiles, f"Exposure Profile (SOURCED DATA, as of {ref_date}) -- {n_paths} paths, {rate_model}/GBM_SV/FXGBM_SV", out_path_png)
    print(f"\nSaved exposure plot to {out_path}")

    factor_fan_factors = [rate_factor] + list(factors.equities[:3]) + list(factors.fx)
    fan_paths = plot_all_factor_fans(precache, factor_fan_factors, OUT_DIR)
    print(f"Saved {len(fan_paths)} factor fan charts to {OUT_DIR}")

    cashflow_paths = plot_all_cashflow_graphs(counterparties, trades, OUT_DIR)
    print(f"Saved {len(cashflow_paths)} cashflow graphs to {OUT_DIR}")

    trade_meta = {
        tid: {"counterparty": trade.counterparty, "type": type(trade).__name__,
              "maturity": result.trade_maturities[tid].isoformat()}
        for tid, trade in trades.items()
    }
    json_payload = {
        "n_paths": n_paths,
        "ref_date": ref_date.isoformat(),
        "horizon": grid.horizon.isoformat(),
        "models": {"rate": rate_model, "equity": "GBM_SV", "fx": "FXGBM_SV"},
        "data_source": "real (FRED + Yahoo Finance) -- see _sourced_book.py",
        "n_correlations_sourced": len(book["market_data"]["correlations"]),
        "trades": trade_meta,
        "profiles": {
            owner_id: {
                "dates": [d.isoformat() for d in p.dates],
                "ee": p.ee,
                "nee": p.nee,
                "pfe_95": p.pfe_95,
                "pfe_99": p.pfe_99,
                "tail_ee_95": p.tail_ee_95,
                "tail_ee_99": p.tail_ee_99,
                "mpe_95": p.mpe_95,
                "mpe_99": p.mpe_99,
                "eepe": p.eepe,
                "tail_eepe_95": p.tail_eepe_95,
                "tail_eepe_99": p.tail_eepe_99,
            }
            for owner_id, p in profiles.items()
        },
    }
    with open(json_out_path, "w") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"Saved data to {json_out_path}")
    return profiles, out_path


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pricing_date = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
    rate_model = sys.argv[4] if len(sys.argv) > 4 else "LGM2F_SV"
    main(n_paths, n_workers, pricing_date, rate_model)
