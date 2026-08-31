"""
Same exposure-profile pipeline as exposure_profile.py, but every market data
input is SOURCED from free public data (FRED, Yahoo Finance) via
risk_engine.market_data instead of a flat placeholder number. This is the
actual MARKET_DATA.md "what you collect" deliverable wired end to end; see
risk_engine/market_data/build_market.py's docstring for exactly which free
source backs each field and what's an honest proxy vs. real data.

Requires internet access (FRED + Yahoo Finance chart API, both free/no key).

    python risk_engine/examples/exposure_profile_sourced.py [n_paths] [n_workers] [ref_date]
    ref_date: ISO format YYYY-MM-DD, defaults to today (sourced data is "live" --
        see the pricing_date parameter's docstring on main() for why the
        valuation date should normally match today, when using real market data).
"""
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

from capitolis_pricers.market import MarketState
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.factors.extract import extract_factors
from risk_engine.market_data.build_market import source_market_data
from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.models.registry import get_rate_model, get_spot_model
from risk_engine.simulation.joint import JointSimulator
from risk_engine.simulation.grid import build_simulation_grid, reporting_anchors, collect_regression_dates
from risk_engine.pricing import price_curves
from risk_engine.netting import build_netting_hierarchy
from risk_engine.exposure import compute_all_profiles
from risk_engine.plotting import plot_exposure_profiles


TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")
EQUITIES_CSV = os.path.join(TRADE_DATA, "underlyings", "equities.csv")
OUT_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(OUT_DIR, "exposure_profile_sourced.png")


def main(n_paths=10_000, n_workers=None, pricing_date=None):
    """pricing_date: the valuation/reference date for this run (date object
    or None -> today). Sourced market data is "live" -- it reflects prices
    observed as of whenever this script runs, so pricing_date should
    normally be left at today's date to stay consistent with when the data
    was actually fetched; passing an earlier/later date reprices the SAME
    book and trades against data sourced today, treated as if observed on
    that other date (a legitimate what-if, not how this script is
    typically meant to be used). Several book trades' start_date can
    predate pricing_date, so some may already be mid-life or past maturity
    as of that date; price_curves' maturity guard (pricing.py) already
    zeros a trade out past its own end_date/forward_date, so a
    shorter-than-full-book exposure profile is real behavior, not a bug.
    """
    REF_DATE = pricing_date or date.today()
    baskets = load_equities(EQUITIES_CSV)
    bonds = load_bonds(os.path.join(TRADE_DATA, "underlyings", "bonds.csv"))
    eqtrs = load_equity_trs(os.path.join(TRADE_DATA, "equity_trs.csv"), baskets)
    bfwd = load_bond_forward(os.path.join(TRADE_DATA, "bond_forward.csv"), bonds)
    btrs = load_bond_trs(os.path.join(TRADE_DATA, "bond_trs.csv"), bonds)
    trades = {**eqtrs, **bfwd, **btrs}

    factors = extract_factors(eqtrs, bfwd, btrs)
    print(f"Factors: {len(factors.rates)} rate, {len(factors.equities)} equity, {len(factors.fx)} FX")

    print("Sourcing market data (FRED + Yahoo Finance)...")
    t0 = time.time()
    market_data = source_market_data(EQUITIES_CSV, REF_DATE, include_fx=bool(factors.fx))
    print(f"  sourcing: {time.time()-t0:.1f}s "
          f"({len(market_data['equity_spot'])}/{len(factors.equities)} equity names fetched)")

    missing = [e.isin for e in factors.equities if e.isin not in market_data["equity_spot"]]
    if missing:
        print(f"  WARNING: {len(missing)} names failed to fetch, will be skipped: {missing}")

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

    # --- rate model: sourced USD curve (real, from FRED); the VOL surface is
    # still a flat placeholder -- no free source publishes a rate vol
    # surface (that's a swaption/cap vol grid, a paid data product), so this
    # one input stays unsourced even in the "sourced data" script. -----------
    rate_calibrated = get_rate_model("LGM2F_SV").calibrate(
        market_data["usd_curve"], flat_vol_surface("RATE_USD", flat_vol=0.010))

    # --- equity models: sourced spot + realized vol per name; dividend rate
    # NOT sourced (see build_market.py) -- left at 0.0, an honest gap, not
    # a fabricated non-zero number --------------------------------------
    sim = JointSimulator()
    rate_factor = factors.rates[0]
    sim.add_rate(rate_factor, rate_calibrated)

    live_equity_factors = [e for e in factors.equities if e.isin in market_data["equity_spot"]]
    for eq_factor in live_equity_factors:
        calibrated = get_spot_model("GBM_SV").calibrate(
            market_data["equity_spot"][eq_factor.isin],
            rate_calibrated,
            market_data["equity_vol_surface"][eq_factor.isin],
            dividend_rate=market_data["equity_dividend_rates"][eq_factor.isin],
        )
        sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)

    # --- FX model: sourced spot, CIP-derived implied JPY curve, realized vol
    if factors.fx and market_data["fx_spot"] is not None:
        fx_factor = factors.fx[0]
        fx_calibrated = get_spot_model("FXGBM_SV").calibrate(
            market_data["fx_spot"], rate_calibrated, market_data["fx_vol_surface"],
            implied_foreign_curve=market_data["implied_jpy_curve"],
        )
        sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor)

    # --- correlations: sourced from realized historical co-movement --------
    # every prior run used MarketState(ref_date=REF_DATE) with NO
    # correlations kwarg, which meant JointSimulator's correlation lookup
    # always defaulted to 0 -- every simulated path was effectively
    # uncorrelated across factors. This is the fix.
    market_state_for_correlation = MarketState(ref_date=REF_DATE, correlations=market_data["correlations"])

    print(f"Precaching {n_paths} paths x {len(grid.dates)} simulation dates "
          f"({len(market_data['correlations'])} sourced pairwise correlations)...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    precache = sim.simulate(market_state_for_correlation, n_paths=n_paths, horizon_dates=grid.dates, rng=rng, ref_date=REF_DATE)
    print(f"  precache: {time.time()-t0:.1f}s")

    live_trade_ids = set(eqtrs) - {tid for tid, t in eqtrs.items()
                                    if any(p.isin not in market_data["equity_spot"] for p in t.positions)}
    priceable_trades = {tid: t for tid, t in trades.items() if tid not in eqtrs or tid in live_trade_ids}
    if len(priceable_trades) < len(trades):
        print(f"  Skipping {len(trades)-len(priceable_trades)} equity TRS trade(s) referencing un-fetched names")

    print(f"Pricing {len(priceable_trades)} trades x {n_paths} paths x {len(regression_dates)} regression dates "
          f"(n_workers={n_workers or os.cpu_count()})...")
    t0 = time.time()
    result = price_curves(priceable_trades, precache, regression_dates, equity_dividend_rates=market_data["equity_dividend_rates"], n_workers=n_workers)
    print(f"  price_curves: {time.time()-t0:.1f}s")

    counterparties = build_netting_hierarchy(priceable_trades)
    print(f"Counterparties: {[c.id for c in counterparties]}")

    profiles = compute_all_profiles(counterparties, result, anchors, REF_DATE)
    for owner_id, p in sorted(profiles.items()):
        print(f"{owner_id:12s} MPE_99 = {p.mpe_99:15,.2f}  MPE_95 = {p.mpe_95:15,.2f}  "
              f"EEPE = {p.eepe:15,.2f}  max EE = {max(p.ee) if p.ee else 0.0:15,.2f}")

    out_path = plot_exposure_profiles(
        profiles, f"Exposure Profile (SOURCED DATA, as of {REF_DATE}) -- {n_paths} paths, LGM2F_SV/GBM_SV/FXGBM_SV", OUT_PATH)
    print(f"\nSaved exposure plot to {out_path}")
    return profiles, out_path


if __name__ == "__main__":
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pricing_date = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None
    main(n_paths, n_workers, pricing_date)
