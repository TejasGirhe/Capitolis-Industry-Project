"""
Price the full 16-trade book under joint LGM2F_SV (rates) + GBM_SV (equity) +
FXGBM_SV (FX) simulation.

    python -m risk_engine.examples.price_book_lgm2fsv
"""
import math
import os
import sys
from datetime import date, timedelta

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs
from capitolis_pricers.curves import zero_curve

from risk_engine.factors.extract import extract_factors
from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.calibration.implied_fx_curve import build_implied_jpy_curve
from risk_engine.models.registry import get_rate_model, get_spot_model
from risk_engine.simulation.joint import JointSimulator
from risk_engine.pricing import price_book


REF_DATE = date(2026, 1, 15)
TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")


def main():
    baskets = load_equities(os.path.join(TRADE_DATA, "underlyings", "equities.csv"))
    bonds = load_bonds(os.path.join(TRADE_DATA, "underlyings", "bonds.csv"))
    eqtrs = load_equity_trs(os.path.join(TRADE_DATA, "equity_trs.csv"), baskets)
    bfwd = load_bond_forward(os.path.join(TRADE_DATA, "bond_forward.csv"), bonds)
    btrs = load_bond_trs(os.path.join(TRADE_DATA, "bond_trs.csv"), bonds)
    trades = {**eqtrs, **bfwd, **btrs}

    factors = extract_factors(eqtrs, bfwd, btrs)
    print(f"Factors: {len(factors.rates)} rate, {len(factors.equities)} equity, {len(factors.fx)} FX")

    usd_curve = zero_curve(REF_DATE, [0.5, 1, 2, 5, 10], [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])
    rate_vol_surface = flat_vol_surface("RATE_USD", flat_vol=0.010)
    rate_calibrated = get_rate_model("LGM2F_SV").calibrate(usd_curve, rate_vol_surface)

    # Illustrative flat spots/vols per ISIN (a real run sources these from
    # collected market data per MARKET_DATA.md Sec.3/5) -- every equity name
    # in the book gets the same placeholder level here purely to demonstrate
    # the pipeline end-to-end.
    equity_spot = {isin: 100.0 for isin in {e.isin for e in factors.equities}}
    equity_div = {isin: 0.015 for isin in equity_spot}

    sim = JointSimulator()
    rate_factor = factors.rates[0]
    sim.add_rate(rate_factor, rate_calibrated)

    for eq_factor in factors.equities:
        vol_surface = flat_vol_surface(str(eq_factor), flat_vol=0.22)
        calibrated = get_spot_model("GBM_SV").calibrate(
            equity_spot[eq_factor.isin], rate_calibrated, vol_surface,
            dividend_rate=equity_div[eq_factor.isin],
        )
        sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)

    if factors.fx:
        fx_factor = factors.fx[0]
        fx_spot = 150.0
        # Illustrative forward points: JPY rates ~1.5% below USD (typical
        # historical USD/JPY differential), so USDJPY forwards trade below
        # spot -- a real run sources these from FX swap points per
        # MARKET_DATA.md Sec.2.2, not this placeholder differential.
        jpy_usd_rate_differential = 0.015
        implied_jpy = build_implied_jpy_curve(
            REF_DATE, fx_spot,
            [(t, fx_spot * math.exp(-jpy_usd_rate_differential * t)) for t in (0.5, 1.0, 2.0)],
            usd_curve,
        )
        fx_vol_surface = flat_vol_surface(str(fx_factor), flat_vol=0.10)
        fx_calibrated = get_spot_model("FXGBM_SV").calibrate(
            fx_spot, rate_calibrated, fx_vol_surface, implied_foreign_curve=implied_jpy,
        )
        sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor)

    horizon_dates = [REF_DATE + timedelta(days=d) for d in (0, 30, 90, 180, 365)]
    rng = np.random.default_rng(42)
    joint_result = sim.simulate(_dummy_market_state(), n_paths=2000, horizon_dates=horizon_dates, rng=rng, ref_date=REF_DATE)

    result = price_book(trades, joint_result, equity_dividend_rates=equity_div, reporting=True)

    print(f"\n{len(trades)} trades priced across {len(result.path_ids)} paths x {len(result.horizon_dates)} horizon dates\n")
    means_t0 = result.mean_npv_by_trade(date_index=0)
    for tid in result.trade_ids:
        print(f"{tid:12s} mean NPV @ t=0 = {means_t0[tid]:15,.2f} USD")


def _dummy_market_state():
    """JointSimulator.simulate() needs a MarketState only for its
    .correlation(a, b) lookups; the sample book ships no correlations
    (MARKET_DATA.md Sec.6 says collect them -- students populate this),
    so an empty MarketState defaults every pair to 0 correlation, matching
    MarketState.correlation's own documented fallback."""
    from capitolis_pricers.market import MarketState
    return MarketState(ref_date=REF_DATE)


if __name__ == "__main__":
    main()
