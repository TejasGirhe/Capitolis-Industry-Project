"""
Shared real-market-data sourcing + simulator-building logic, extracted from
exposure_profile_sourced.py so exposure_profile.py, greeks_report.py, and
xva_report.py all wire the SAME real-data path (FRED USD curve, Yahoo equity/
FX spot + realized vol, realized correlation) instead of each hand-rolling
flat placeholders independently. One place this wiring lives, not four.

    python risk_engine/examples/exposure_profile_sourced.py [n_paths] [n_workers] [ref_date]
    (that script is now a thin wrapper around build_sourced_book(), kept as
    the standalone entry point it always was)
"""
import os
import sys
import time
from datetime import date

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

TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")
EQUITIES_CSV = os.path.join(TRADE_DATA, "underlyings", "equities.csv")

BASE_RATE_VOL = 0.010  # rate vol has no free source -- stays a flat placeholder even here


def build_sourced_book(ref_date=None, rate_model="LGM2F_SV"):
    """Loads the book, sources real market data (FRED + Yahoo), and returns
    everything needed to run the exposure/greeks/xva pipelines against it.

    ref_date: None -> date.today() (sourced data is "live" -- valuation date
        should match when it was observed; confirmed design, see
        exposure_profile_sourced.py's original docstring). Passing an
        explicit date reprices the SAME sourced data as-of a different
        valuation date -- a legitimate what-if, not the default use.

    rate_model: registry name of the rate model, default "LGM2F_SV" (the
        production choice). Pass e.g. "LGM1F" for a one-off model-comparison
        run -- swaps the model everywhere this book's simulator uses rates
        (base case AND the vega rate-bump path), nothing else changes.

    Returns a dict:
        ref_date, trades (only trades fully priceable with fetched data),
        all_trades (unfiltered, for reference), factors, market_data,
        grid, anchors, regression_dates, rate_calibrated,
        build_simulator(vol_bump_by_group=None) -> JointSimulator,
        market_state_for_correlation (MarketState carrying the sourced
        correlation matrix -- pass this, not a bare MarketState, into
        JointSimulator.simulate()).

    build_simulator's vol_bump_by_group: optional {"rate": x, "equity": y,
    "fx": z} dict of ADDITIVE vol bumps, applied on top of the sourced
    vol level for that factor group -- lets greeks_report.py's vega
    bump-and-reprice reuse this SAME builder instead of hand-rolling its
    own closure (previously duplicated logic).
    """
    REF_DATE = ref_date or date.today()

    baskets = load_equities(EQUITIES_CSV)
    bonds = load_bonds(os.path.join(TRADE_DATA, "underlyings", "bonds.csv"))
    eqtrs = load_equity_trs(os.path.join(TRADE_DATA, "equity_trs.csv"), baskets)
    bfwd = load_bond_forward(os.path.join(TRADE_DATA, "bond_forward.csv"), bonds)
    btrs = load_bond_trs(os.path.join(TRADE_DATA, "bond_trs.csv"), bonds)
    all_trades = {**eqtrs, **bfwd, **btrs}

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

    live_trade_ids = set(eqtrs) - {tid for tid, t in eqtrs.items()
                                    if any(p.isin not in market_data["equity_spot"] for p in t.positions)}
    trades = {tid: t for tid, t in all_trades.items() if tid not in eqtrs or tid in live_trade_ids}
    if len(trades) < len(all_trades):
        print(f"  Skipping {len(all_trades)-len(trades)} equity TRS trade(s) referencing un-fetched names")

    max_maturity = max(getattr(t, "end_date", None) or t.forward_date for t in trades.values())
    grid = build_simulation_grid(REF_DATE, max_maturity)
    anchors = reporting_anchors(REF_DATE, max_maturity)
    regression_dates = collect_regression_dates(trades, anchors, ref_date=REF_DATE)
    print(f"Simulation grid: horizon {grid.horizon}, {len(grid.dates)} dates")
    print(f"Regression dates: {len(regression_dates)}, Reporting anchors: {len(anchors)}")

    rate_calibrated = get_rate_model(rate_model).calibrate(
        market_data["usd_curve"], flat_vol_surface("RATE_USD", flat_vol=BASE_RATE_VOL))

    live_equity_factors = [e for e in factors.equities if e.isin in market_data["equity_spot"]]

    def build_simulator(vol_bump_by_group: dict = None) -> JointSimulator:
        bump = vol_bump_by_group or {}
        rate_bump = bump.get("rate", 0.0)
        equity_bump = bump.get("equity", 0.0)
        fx_bump = bump.get("fx", 0.0)

        rc = rate_calibrated if rate_bump == 0.0 else get_rate_model(rate_model).calibrate(
            market_data["usd_curve"], flat_vol_surface("RATE_USD", flat_vol=BASE_RATE_VOL + rate_bump))

        sim = JointSimulator()
        rate_factor = factors.rates[0]
        sim.add_rate(rate_factor, rc)

        for eq_factor in live_equity_factors:
            vol_surface = market_data["equity_vol_surface"][eq_factor.isin]
            if equity_bump != 0.0:
                from risk_engine.calibration.market_surface import VolSurface
                vol_surface = VolSurface(
                    factor_key=vol_surface.factor_key, tenors=vol_surface.tenors, strikes=vol_surface.strikes,
                    vols={k: v + equity_bump for k, v in vol_surface.vols.items()},
                )
            calibrated = get_spot_model("GBM_SV").calibrate(
                market_data["equity_spot"][eq_factor.isin], rc, vol_surface,
                dividend_rate=market_data["equity_dividend_rates"][eq_factor.isin])
            sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)

        if factors.fx and market_data["fx_spot"] is not None:
            fx_factor = factors.fx[0]
            fx_vol_surface = market_data["fx_vol_surface"]
            if fx_bump != 0.0:
                from risk_engine.calibration.market_surface import VolSurface
                fx_vol_surface = VolSurface(
                    factor_key=fx_vol_surface.factor_key, tenors=fx_vol_surface.tenors, strikes=fx_vol_surface.strikes,
                    vols={k: v + fx_bump for k, v in fx_vol_surface.vols.items()},
                )
            fx_calibrated = get_spot_model("FXGBM_SV").calibrate(
                market_data["fx_spot"], rc, fx_vol_surface, implied_foreign_curve=market_data["implied_jpy_curve"])
            sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor)

        return sim

    market_state_for_correlation = MarketState(ref_date=REF_DATE, correlations=market_data["correlations"])

    return {
        "ref_date": REF_DATE,
        "trades": trades,
        "all_trades": all_trades,
        "factors": factors,
        "market_data": market_data,
        "grid": grid,
        "anchors": anchors,
        "regression_dates": regression_dates,
        "rate_calibrated": rate_calibrated,
        "build_simulator": build_simulator,
        "market_state_for_correlation": market_state_for_correlation,
    }
