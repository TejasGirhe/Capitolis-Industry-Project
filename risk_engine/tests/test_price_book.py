"""
End-to-end check: price_curves against the real 16-trade book under joint
LGM2F_SV (rates) + GBM_SV (equity) + FXGBM_SV (FX) simulation, via the
3-stage precache/interpolate/aggregate pipeline.

The key correctness property tested here is the t=ref_date zero-variance
collapse: at the valuation date itself the interpolated state is the
simulation's own t=0 state (exactly zero, no time elapsed to diffuse), so
NPV0 at ref_date across all paths must reproduce today's deterministic NPV
computed directly against a plain MarketState (the same snapshot
examples/price_all.py prices) -- this is what actually proves the
precache -> interpolate -> price bridge (state_at, build_market_states_at)
is wired correctly end to end, not just that the code runs without raising.
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pytest

RISK_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICERS_ROOT = os.path.join(os.path.dirname(RISK_ENGINE_ROOT), "capitolis_pricers", "capitolis_pricers")
for p in (RISK_ENGINE_ROOT, PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from capitolis_pricers.market import MarketState
from capitolis_pricers.curves import zero_curve, FxCurve
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.factors.extract import extract_factors
from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.calibration.implied_fx_curve import build_implied_jpy_curve
from risk_engine.models.registry import get_rate_model, get_spot_model
from risk_engine.simulation.joint import JointSimulator
from risk_engine.simulation.grid import build_simulation_grid, reporting_anchors, collect_regression_dates
from risk_engine.pricing import price_curves


REF_DATE = date(2026, 1, 15)
TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")
FLAT_SPOT = 100.0
FLAT_DIV = 0.015
FX_SPOT = 150.0


@pytest.fixture(scope="module")
def book():
    baskets = load_equities(os.path.join(TRADE_DATA, "underlyings", "equities.csv"))
    bonds = load_bonds(os.path.join(TRADE_DATA, "underlyings", "bonds.csv"))
    eqtrs = load_equity_trs(os.path.join(TRADE_DATA, "equity_trs.csv"), baskets)
    bfwd = load_bond_forward(os.path.join(TRADE_DATA, "bond_forward.csv"), bonds)
    btrs = load_bond_trs(os.path.join(TRADE_DATA, "bond_trs.csv"), bonds)
    return {**eqtrs, **bfwd, **btrs}


@pytest.fixture(scope="module")
def usd_curve():
    return zero_curve(REF_DATE, [0.5, 1, 2, 5, 10], [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])


def _max_maturity(book):
    return max(t.end_date if hasattr(t, "end_date") else t.forward_date for t in book.values())


def test_npv0_at_ref_date_matches_deterministic_pricing(book, usd_curve):
    factors = extract_factors(
        {k: v for k, v in book.items() if k.startswith("EQTRS")},
        {k: v for k, v in book.items() if k.startswith("BF")},
        {k: v for k, v in book.items() if k.startswith("BTRS")},
    )
    grid = build_simulation_grid(REF_DATE, _max_maturity(book))
    rate_calibrated = get_rate_model("LGM2F_SV").calibrate(usd_curve, flat_vol_surface("RATE_USD", flat_vol=0.010))

    equity_spot = {isin: FLAT_SPOT for isin in {e.isin for e in factors.equities}}
    equity_div = {isin: FLAT_DIV for isin in equity_spot}

    sim = JointSimulator()
    rate_factor = factors.rates[0]
    sim.add_rate(rate_factor, rate_calibrated)

    for eq_factor in factors.equities:
        vol_surface = flat_vol_surface(str(eq_factor), flat_vol=0.22)
        calibrated = get_spot_model("GBM_SV").calibrate(
            equity_spot[eq_factor.isin], rate_calibrated, vol_surface, dividend_rate=equity_div[eq_factor.isin])
        sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)

    implied_jpy = build_implied_jpy_curve(REF_DATE, FX_SPOT, [(t, FX_SPOT) for t in (0.5, 1.0, 2.0)], usd_curve)
    fx_factor = factors.fx[0]
    fx_calibrated = get_spot_model("FXGBM_SV").calibrate(
        FX_SPOT, rate_calibrated, flat_vol_surface(str(fx_factor), flat_vol=0.10), implied_foreign_curve=implied_jpy)
    sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor)

    # n_paths=5 is enough here: t=0 state is now bit-exact (see the dt-floor
    # fix in models/lgm.py|gbm.py|fx.py's simulate_paths -- previously a
    # spurious 1e-10 dt floor at t=0 injected real MC noise, which is why
    # this test used to need hundreds of paths and ~9 minutes to average
    # out). Every path gives the identical exact-zero state at t=0, so a
    # handful of paths proves the same thing as 500 in a fraction of the time.
    rng = np.random.default_rng(11)
    precache = sim.simulate(MarketState(ref_date=REF_DATE), n_paths=5, horizon_dates=grid.dates,
                             rng=rng, ref_date=REF_DATE)

    anchors = reporting_anchors(REF_DATE, _max_maturity(book))
    regression_dates = collect_regression_dates(book, anchors, ref_date=REF_DATE)
    result = price_curves(book, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=1)

    mean_npv0 = result.mean_npv0_by_trade(REF_DATE)

    fx_curve = FxCurve("USD", "JPY", FX_SPOT)
    isins = equity_spot.keys()
    reference_market = MarketState(
        ref_date=REF_DATE, discount_curves={"USD": usd_curve},
        equity_spots={i: FLAT_SPOT for i in isins}, equity_dividend_rates=equity_div,
        fx_curves={("USD", "JPY"): fx_curve},
    )
    for tid, trade in book.items():
        expected = trade.npv(reference_market, reporting=True)
        # Tight tolerance: t=0 state is bit-exact zero (see dt-floor fix
        # note above), so NPV0 at ref_date should match the deterministic
        # reference to near machine precision, not just within MC noise.
        assert mean_npv0[tid] == pytest.approx(expected, rel=1e-9, abs=1e-6), (
            f"{tid}: NPV0 at ref_date mean {mean_npv0[tid]:.2f} vs deterministic {expected:.2f}"
        )


def test_price_curves_produces_finite_npvs(book, usd_curve):
    """Sanity: no NaN/inf anywhere in NPV0/NPV10, including a real future date."""
    factors = extract_factors(
        {k: v for k, v in book.items() if k.startswith("EQTRS")},
        {k: v for k, v in book.items() if k.startswith("BF")},
        {k: v for k, v in book.items() if k.startswith("BTRS")},
    )
    grid = build_simulation_grid(REF_DATE, _max_maturity(book))
    rate_calibrated = get_rate_model("LGM1F").calibrate(usd_curve, flat_vol_surface("RATE_USD", flat_vol=0.01))
    sim = JointSimulator()
    rate_factor = factors.rates[0]
    sim.add_rate(rate_factor, rate_calibrated)
    equity_div = {}
    for eq_factor in factors.equities:
        calibrated = get_spot_model("GBM").calibrate(
            100.0, rate_calibrated, flat_vol_surface(str(eq_factor), flat_vol=0.22), dividend_rate=0.01)
        equity_div[eq_factor.isin] = 0.01
        sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)
    implied_jpy = build_implied_jpy_curve(REF_DATE, 150.0, [(t, 150.0) for t in (0.5, 1.0, 2.0)], usd_curve)
    fx_factor = factors.fx[0]
    fx_calibrated = get_spot_model("FXGBM").calibrate(
        150.0, rate_calibrated, flat_vol_surface(str(fx_factor), flat_vol=0.10), implied_foreign_curve=implied_jpy)
    sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor)

    rng = np.random.default_rng(3)
    precache = sim.simulate(MarketState(ref_date=REF_DATE), n_paths=200, horizon_dates=grid.dates,
                             rng=rng, ref_date=REF_DATE)
    anchors = reporting_anchors(REF_DATE, _max_maturity(book))
    regression_dates = collect_regression_dates(book, anchors, ref_date=REF_DATE)
    result = price_curves(book, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=1)
    for v in result.npv0.values():
        assert np.isfinite(v)
    for v in result.npv10.values():
        assert np.isfinite(v)
