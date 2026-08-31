"""
Per-trade INDEPENDENT simulation exposure profile -- an alternative to
exposure_profile.py's book-wide joint simulation. Each trade gets its own
small Cholesky decomposition over ONLY its own factors and its own
independent random draw (risk_engine.simulation.per_trade), confirmed
explicitly with the user as a deliberate tradeoff: exposure numbers
computed this way do NOT reflect cross-trade correlation, unlike
exposure_profile.py's netting-set-aware numbers.

Per-trade exposure profiles (EE/PFE_95/PFE_99/MPE/EEPE) are directly valid
-- a single trade has no cross-trade correlation to lose. Counterparty-level
numbers here are an INDEPENDENT-TRADE AGGREGATE (sum of each trade's own
marginal EE/PFE, NOT a netted/correlated number -- see
risk_engine.netting.independent_aggregate's module docstring for exactly
what this over/understates and why), clearly labeled as such in every
printout and plot.

    python risk_engine/examples/exposure_profile_per_trade.py [n_paths] [n_workers]
"""
import json
import os
import sys
import time
from datetime import date

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from capitolis_pricers.curves import zero_curve
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.simulation.grid import build_simulation_grid, reporting_anchors, collect_regression_dates
from risk_engine.simulation.per_trade import simulate_all_trades_independently
from risk_engine.pricing import price_curves_per_trade
from risk_engine.exposure import compute_per_trade_profiles
from risk_engine.netting import aggregate_all_independent_profiles


REF_DATE = date(2026, 1, 15)
TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")
OUT_DIR = os.path.dirname(__file__)
JSON_OUT_PATH = os.path.join(OUT_DIR, "exposure_profile_per_trade.json")


def main(n_paths=10_000, n_workers=None):
    baskets = load_equities(os.path.join(TRADE_DATA, "underlyings", "equities.csv"))
    bonds = load_bonds(os.path.join(TRADE_DATA, "underlyings", "bonds.csv"))
    eqtrs = load_equity_trs(os.path.join(TRADE_DATA, "equity_trs.csv"), baskets)
    bfwd = load_bond_forward(os.path.join(TRADE_DATA, "bond_forward.csv"), bonds)
    btrs = load_bond_trs(os.path.join(TRADE_DATA, "bond_trs.csv"), bonds)
    trades = {**eqtrs, **bfwd, **btrs}
    print(f"{len(trades)} trades, each simulated INDEPENDENTLY (own Cholesky, own rng)")

    max_maturity = max(
        [t.end_date for t in eqtrs.values()]
        + [t.forward_date for t in bfwd.values()]
        + [t.end_date for t in btrs.values()]
    )
    grid = build_simulation_grid(REF_DATE, max_maturity)
    anchors = reporting_anchors(REF_DATE, max_maturity)
    regression_dates = collect_regression_dates(trades, anchors, ref_date=REF_DATE)
    print(f"Simulation grid: horizon {grid.horizon}, {len(grid.dates)} dates")
    print(f"Regression dates: {len(regression_dates)}, Reporting anchors: {len(anchors)}")

    usd_curve = zero_curve(REF_DATE, [0.5, 1, 2, 5, 10], [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])
    rate_vol = flat_vol_surface("RATE_USD", flat_vol=0.010)
    all_isins = {p.isin for t in trades.values() if hasattr(t, "positions") for p in t.positions}
    equity_spot = {isin: 100.0 for isin in all_isins}
    equity_vol_surface = {isin: flat_vol_surface(f"EQ_{isin}", flat_vol=0.22) for isin in all_isins}
    equity_div = {isin: 0.015 for isin in all_isins}
    fx_spot = 150.0
    fx_vol_surface = flat_vol_surface("FX_USDJPY", flat_vol=0.10)
    implied_jpy = None
    if any(any(p.currency != t.trade_currency for p in t.positions) for t in eqtrs.values()):
        from risk_engine.calibration.implied_fx_curve import build_implied_jpy_curve
        implied_jpy = build_implied_jpy_curve(REF_DATE, fx_spot, [(t, fx_spot) for t in (0.5, 1.0, 2.0)], usd_curve)

    print(f"\nSimulating {len(trades)} independent trades x {n_paths} paths x {len(grid.dates)} dates...")
    t0 = time.time()
    precache_by_trade = simulate_all_trades_independently(
        trades, n_paths, grid.dates, REF_DATE, "LGM2F_SV", "GBM_SV", "FXGBM_SV",
        usd_curve, rate_vol, equity_spot, equity_vol_surface, equity_div,
        fx_spot=fx_spot, fx_vol_surface=fx_vol_surface, implied_foreign_curve=implied_jpy)
    print(f"  independent precache: {time.time()-t0:.1f}s")

    print(f"Pricing each trade against its own independent precache (n_workers={n_workers or os.cpu_count()})...")
    t0 = time.time()
    curves_by_trade = price_curves_per_trade(trades, precache_by_trade, regression_dates, equity_div, n_workers=n_workers)
    print(f"  price_curves_per_trade: {time.time()-t0:.1f}s")

    profiles = compute_per_trade_profiles(trades, curves_by_trade, anchors, REF_DATE)
    print("\nPer-trade exposure (independent, no cross-trade correlation):")
    for tid, p in sorted(profiles.items()):
        print(f"  {tid:12s} MPE_99={p.mpe_99:14,.2f}  MPE_95={p.mpe_95:14,.2f}  "
              f"EEPE={p.eepe:14,.2f}  max EE={max(p.ee) if p.ee else 0.0:14,.2f}")

    aggregates = aggregate_all_independent_profiles(trades, profiles)
    print("\nCounterparty-level INDEPENDENT AGGREGATES (NOT netted/correlated -- see label):")
    for cpty, agg in sorted(aggregates.items()):
        print(f"  {agg.label()}")
        print(f"    MPE_agg_99={agg.mpe_aggregate_99:14,.2f}  MPE_agg_95={agg.mpe_aggregate_95:14,.2f}  "
              f"max EE_agg={max(agg.ee_aggregate) if agg.ee_aggregate else 0.0:14,.2f}")

    json_payload = {
        "mode": "per_trade_independent",
        "n_paths": n_paths,
        "ref_date": REF_DATE.isoformat(),
        "per_trade_profiles": {
            tid: {
                "dates": [d.isoformat() for d in p.dates], "ee": p.ee,
                "pfe_95": p.pfe_95, "pfe_99": p.pfe_99, "mpe_95": p.mpe_95, "mpe_99": p.mpe_99, "eepe": p.eepe,
            }
            for tid, p in profiles.items()
        },
        "counterparty_independent_aggregates": {
            cpty: {
                "label": agg.label(), "dates": [d.isoformat() for d in agg.dates],
                "ee_aggregate": agg.ee_aggregate, "pfe_aggregate_95": agg.pfe_aggregate_95,
                "pfe_aggregate_99": agg.pfe_aggregate_99, "mpe_aggregate_95": agg.mpe_aggregate_95,
                "mpe_aggregate_99": agg.mpe_aggregate_99,
            }
            for cpty, agg in aggregates.items()
        },
    }
    with open(JSON_OUT_PATH, "w") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"\nSaved data to {JSON_OUT_PATH}")
    return profiles, aggregates


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(n_paths, n_workers)
