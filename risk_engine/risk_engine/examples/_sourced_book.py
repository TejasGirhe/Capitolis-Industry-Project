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
import pickle
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
from risk_engine.factors.types import RateFactor
from risk_engine.market_data.build_market import source_market_data
from risk_engine.calibration.market_surface import flat_vol_surface, VolSurface
from risk_engine.models.registry import get_rate_model, get_spot_model
from risk_engine.simulation.joint import JointSimulator
from risk_engine.simulation.grid import build_simulation_grid, reporting_anchors, collect_regression_dates

TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")
EQUITIES_CSV = os.path.join(TRADE_DATA, "underlyings", "equities.csv")

BASE_RATE_VOL = 0.010  # fallback ONLY -- used if NEITHER the real Bloomberg USD swaption
                        # cube NOR the Databento SOFR-options proxy is available (see
                        # _rate_vol_surface below) -- last-resort placeholder, not the
                        # default path under normal conditions.

RAW_DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results", "_raw_data_cache")


def _raw_cache_path(ref_date):
    return os.path.join(RAW_DATA_CACHE_DIR, f"raw_sourced_data_{ref_date.isoformat()}.pkl")


def _load_or_fetch_raw_data(ref_date, factors, rate_model):
    """Caches ONLY the live-fetched/loaded raw inputs (FRED+Yahoo market_data,
    Bloomberg USD/JPY curves+vol surfaces) to a local pickle keyed by
    ref_date -- NOT the calibrated models or JointSimulator (those are
    closed-form/cheap to rebuild locally per docstring note above, and
    calibrated-model objects are not guaranteed picklable/stable across
    processes). Lets many concurrent OS-process trials (see
    mc_convergence_analysis.py's parallel launcher) share ONE live data
    fetch instead of each re-fetching independently -- avoids redundant
    ~5.5min FRED/Yahoo calls and simultaneous-request rate-limit risk.
    A simple file-existence check (no locking): if two processes race to
    fetch, at worst one fetch is wasted and overwritten -- never corrupts
    a partial file, since writes go to a .tmp path and are renamed
    atomically only after the fetch completes.
    """
    cache_path = _raw_cache_path(ref_date)
    if os.path.exists(cache_path):
        print(f"  Using cached raw market data from {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print("Sourcing market data (FRED + Yahoo Finance)...")
    t0 = time.time()
    market_data = source_market_data(EQUITIES_CSV, ref_date, include_fx=bool(factors.fx))
    print(f"  sourcing: {time.time()-t0:.1f}s "
          f"({len(market_data['equity_spot'])}/{len(factors.equities)} equity names fetched)")

    market_data["usd_curve"] = _usd_curve(market_data["usd_curve"])
    base_rate_vol_surface = _rate_vol_surface(ref_date)
    jpy_curve, jpy_vol_surface = _jpy_curve_and_vol(ref_date)

    try:
        from risk_engine.market_data.bloomberg_data import equity_vol_scaling_factor, is_available
        if not is_available():
            raise RuntimeError("data_bloomberg/ directory not found")
        vol_scale_by_ccy = {"USD": equity_vol_scaling_factor("SPX"), "JPY": equity_vol_scaling_factor("TOPIX")}
    except Exception:
        vol_scale_by_ccy = {"USD": 1.0, "JPY": 1.0}

    raw = {
        "market_data": market_data, "base_rate_vol_surface": base_rate_vol_surface,
        "jpy_curve": jpy_curve, "jpy_vol_surface": jpy_vol_surface, "vol_scale_by_ccy": vol_scale_by_ccy,
    }
    os.makedirs(RAW_DATA_CACHE_DIR, exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(raw, fh)
    os.replace(tmp, cache_path)
    return raw


JPY_RATE_FACTOR = RateFactor(currency="JPY")   # NOT derived from trade extraction (this book
    # has no JPY-currency-denominated cashflow -- the JPY equity basket is a USD-settled FX
    # compo, not JPY discounting) -- added explicitly, purely to drive the USD/JPY FX
    # forward's rate differential with a REAL simulated factor instead of a deterministic
    # implied curve (see models/fx.py's module docstring for the full rationale).


def _shifted_surface(surface: VolSurface, bump: float) -> VolSurface:
    """A copy of `surface` with `bump` added to every (tenor, strike) vol --
    used for the vega bump-and-reprice scenario, which needs the SAME
    surface shape (real smile+term-structure, or the flat fallback) shifted
    uniformly, not replaced with a flat bumped placeholder regardless of
    which base surface was actually used."""
    return VolSurface(factor_key=surface.factor_key, tenors=list(surface.tenors),
                      strikes=list(surface.strikes),
                      vols={k: v + bump for k, v in surface.vols.items()})


def _usd_curve(fred_usd_curve):
    """Real USD discount curve -- PRIMARY source is the Bloomberg SWPM zero-
    curve export (data_bloomberg/rates/usd_sofr_bloomberg_zero_discount_
    curve_2026-08-31.csv), which already has discount factors computed by
    Bloomberg directly (no bootstrapping needed, dealer-quality curve) --
    confirmed as the preferred source over this project's own FRED-pillar
    bootstrap. Falls back to the FRED curve (fred_usd_curve, already
    sourced by market_data.build_market.source_market_data) if the
    Bloomberg data drop is unavailable -- printed explicitly, not silent."""
    try:
        from risk_engine.market_data.bloomberg_data import load_usd_curve, is_available
        if not is_available():
            raise RuntimeError("data_bloomberg/ directory not found")
        return load_usd_curve(fred_usd_curve.ref_date)
    except Exception as e:
        print(f"  WARNING: real Bloomberg USD curve unavailable ({e!r}); falling back to FRED-bootstrapped curve")
        return fred_usd_curve


def _rate_vol_surface(ref_date) -> VolSurface:
    """Real USD rate-vol surface (term structure + smile). PRIMARY source
    is now the real USD SOFR ATM normal swaption cube (data_bloomberg/
    rates/usd_sofr_atm_normal_swaption_vol_2026-08-31_*.csv) -- the actual
    OTC instrument this project's rate models need vol from. Falls back to
    the CME SOFR-futures-options Bachelier proxy via Databento
    (market_data/databento_rates.py -- built before the real Bloomberg
    swaption cube was available) if the Bloomberg data drop is missing,
    and to the flat BASE_RATE_VOL placeholder only if BOTH real sources
    fail -- every fallback step printed explicitly, never silent."""
    try:
        from risk_engine.market_data.bloomberg_data import load_usd_swaption_vol_surface, is_available
        if not is_available():
            raise RuntimeError("data_bloomberg/ directory not found")
        return load_usd_swaption_vol_surface()
    except Exception as e:
        print(f"  WARNING: real Bloomberg USD swaption vol cube unavailable ({e!r}); "
              f"trying Databento SOFR-futures-options proxy")
    try:
        from risk_engine.market_data.databento_rates import fetch_sofr_option_smile
        return fetch_sofr_option_smile(ref_date)
    except Exception as e:
        print(f"  WARNING: Databento SOFR-options rate-vol surface also unavailable ({e!r}); "
              f"falling back to flat {BASE_RATE_VOL:.2%} placeholder")
        return flat_vol_surface("RATE_USD", flat_vol=BASE_RATE_VOL)


def _jpy_curve_and_vol(ref_date):
    """Real JPY OIS discount curve + ATM normal swaption vol surface, from
    the Bloomberg data drop -- the FIRST real JPY curve/vol source this
    project has ever had (jpy_rate.py's flat placeholder predates this).
    Returns (None, None) if the Bloomberg data drop is unavailable --
    callers must then fall back to the deterministic implied-curve FX
    drift path (models/fx.py's implied_foreign_curve), not fabricate a
    fake JPY curve/vol."""
    try:
        from risk_engine.market_data.bloomberg_data import load_jpy_curve, load_jpy_swaption_vol_surface, is_available
        if not is_available():
            raise RuntimeError("data_bloomberg/ directory not found")
        return load_jpy_curve(ref_date), load_jpy_swaption_vol_surface()
    except Exception as e:
        print(f"  WARNING: real Bloomberg JPY curve/vol unavailable ({e!r}); "
              f"FX drift will use the deterministic implied-JPY-curve fallback")
        return None, None


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

    raw = _load_or_fetch_raw_data(REF_DATE, factors, rate_model)
    market_data = raw["market_data"]

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

    # market_data["usd_curve"] and base_rate_vol_surface/jpy_curve/jpy_vol_surface
    # are already the REAL (Bloomberg-preferred, FRED/flat-fallback) sourced
    # values -- computed once inside _load_or_fetch_raw_data and cached, not
    # re-fetched here.
    base_rate_vol_surface = raw["base_rate_vol_surface"]
    jpy_curve, jpy_vol_surface = raw["jpy_curve"], raw["jpy_vol_surface"]
    rate_calibrated = get_rate_model(rate_model).calibrate(market_data["usd_curve"], base_rate_vol_surface)
    jpy_rate_calibrated = (get_rate_model(rate_model).calibrate(jpy_curve, jpy_vol_surface)
                            if jpy_curve is not None else None)
    if jpy_rate_calibrated is not None:
        print(f"  Real JPY rate factor calibrated ({rate_model}) off Bloomberg JPY OIS curve + swaption vol")

    live_equity_factors = [e for e in factors.equities if e.isin in market_data["equity_spot"]]

    # Equity vol scaling: real SPX/TOPIX implied-vol / realized-vol ratio
    # applied uniformly to every single-name's realized vol in that
    # index's currency bucket (confirmed design -- a market-level implied-
    # risk-premium adjustment, not a per-name beta regression; see
    # market_data/bloomberg_data.py's equity_vol_scaling_factor docstring
    # for why a raw index vol is NOT substituted directly for single-name
    # vol). US-listed names (native_ccy == 'USD') scaled by the SPX ratio,
    # JPY-listed names by the TOPIX ratio. Falls back to no scaling
    # (ratio=1.0) if the Bloomberg data drop is unavailable.
    vol_scale_by_ccy = raw["vol_scale_by_ccy"]
    if vol_scale_by_ccy != {"USD": 1.0, "JPY": 1.0}:
        print(f"  Equity vol scaling (real SPX/TOPIX implied/realized ratio): USD x{vol_scale_by_ccy['USD']:.4f}, "
              f"JPY x{vol_scale_by_ccy['JPY']:.4f}")
    else:
        print("  WARNING: real equity vol scaling unavailable; using unscaled realized vol")

    def build_simulator(vol_bump_by_group: dict = None) -> JointSimulator:
        bump = vol_bump_by_group or {}
        rate_bump = bump.get("rate", 0.0)
        equity_bump = bump.get("equity", 0.0)
        fx_bump = bump.get("fx", 0.0)

        rc = rate_calibrated if rate_bump == 0.0 else get_rate_model(rate_model).calibrate(
            market_data["usd_curve"], _shifted_surface(base_rate_vol_surface, rate_bump))
        jpy_rc = jpy_rate_calibrated if (rate_bump == 0.0 or jpy_rate_calibrated is None) else get_rate_model(rate_model).calibrate(
            jpy_curve, _shifted_surface(jpy_vol_surface, rate_bump))

        sim = JointSimulator()
        rate_factor = factors.rates[0]
        sim.add_rate(rate_factor, rc)
        if jpy_rc is not None:
            sim.add_rate(JPY_RATE_FACTOR, jpy_rc)

        for eq_factor in live_equity_factors:
            vol_surface = market_data["equity_vol_surface"][eq_factor.isin]
            vol_scale = vol_scale_by_ccy.get(eq_factor.native_ccy, 1.0)
            if vol_scale != 1.0 or equity_bump != 0.0:
                vol_surface = VolSurface(
                    factor_key=vol_surface.factor_key, tenors=vol_surface.tenors, strikes=vol_surface.strikes,
                    vols={k: v * vol_scale + equity_bump for k, v in vol_surface.vols.items()},
                )
            calibrated = get_spot_model("GBM_SV").calibrate(
                market_data["equity_spot"][eq_factor.isin], rc, vol_surface,
                dividend_rate=market_data["equity_dividend_rates"][eq_factor.isin])
            sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)

        if factors.fx and market_data["fx_spot"] is not None:
            fx_factor = factors.fx[0]
            fx_vol_surface = market_data["fx_vol_surface"]
            if fx_bump != 0.0:
                fx_vol_surface = VolSurface(
                    factor_key=fx_vol_surface.factor_key, tenors=fx_vol_surface.tenors, strikes=fx_vol_surface.strikes,
                    vols={k: v + fx_bump for k, v in fx_vol_surface.vols.items()},
                )
            if jpy_rc is not None:
                # Real simulated JPY factor drives the foreign leg -- see
                # models/fx.py's module docstring. implied_foreign_curve is
                # NOT passed here (foreign_rate_model takes precedence, and
                # passing both would be misleading about which is active).
                fx_calibrated = get_spot_model("FXGBM_SV").calibrate(
                    market_data["fx_spot"], rc, fx_vol_surface, foreign_rate_model=jpy_rc)
                sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor,
                            foreign_rate_factor=JPY_RATE_FACTOR)
            else:
                # Fallback: deterministic implied-JPY-curve drift (pre-real-JPY-data path)
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
