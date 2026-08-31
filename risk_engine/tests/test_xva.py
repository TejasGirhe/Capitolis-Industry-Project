"""
xVA (CVA/DVA/FVA) tests: CreditCurve sourcing/tiering, NEE sanity, and the
CVA/credit-tier interaction (better credit -> lower CVA per unit of EE),
against a small, fast simulation of the real book (not the full 10,000-path
production scale -- matches this session's established test-scale pattern).
"""
import os
import sys
from datetime import date

import numpy as np
import pytest

RISK_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICERS_ROOT = os.path.join(os.path.dirname(RISK_ENGINE_ROOT), "capitolis_pricers", "capitolis_pricers")
for p in (RISK_ENGINE_ROOT, PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from capitolis_pricers.market import MarketState
from capitolis_pricers.curves import zero_curve
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.factors.extract import extract_factors
from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.calibration.implied_fx_curve import build_implied_jpy_curve
from risk_engine.models.registry import get_rate_model, get_spot_model
from risk_engine.simulation.joint import JointSimulator
from risk_engine.simulation.grid import build_simulation_grid, reporting_anchors, collect_regression_dates
from risk_engine.pricing import price_curves
from risk_engine.netting import build_netting_hierarchy
from risk_engine.exposure import compute_all_profiles
from risk_engine.market_data.credit import (
    fetch_rating_tier_spread, book_size_by_counterparty, spread_for_counterparty,
    build_credit_curves, build_own_credit_curve,
)
from risk_engine.xva import compute_all_xva


REF_DATE = date(2026, 1, 15)
TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")


@pytest.fixture(scope="module")
def book():
    baskets = load_equities(os.path.join(TRADE_DATA, "underlyings", "equities.csv"))
    bonds = load_bonds(os.path.join(TRADE_DATA, "underlyings", "bonds.csv"))
    eqtrs = load_equity_trs(os.path.join(TRADE_DATA, "equity_trs.csv"), baskets)
    bfwd = load_bond_forward(os.path.join(TRADE_DATA, "bond_forward.csv"), bonds)
    btrs = load_bond_trs(os.path.join(TRADE_DATA, "bond_trs.csv"), bonds)
    return {**eqtrs, **bfwd, **btrs}


# ---------------------------------------------------------------- credit tiering (real FRED data)

def test_book_size_ranking_direction(book):
    sizes = book_size_by_counterparty(book)
    assert sizes["CPTY_C"] > sizes["CPTY_A"] > sizes["CPTY_B"]


def test_larger_book_gets_lower_spread(book):
    aaa = fetch_rating_tier_spread("AAA")
    baa = fetch_rating_tier_spread("BAA")
    assert aaa < baa  # AAA is the better (lower-spread) tier

    spread_c = spread_for_counterparty(book, "CPTY_C", aaa, baa)  # largest book
    spread_a = spread_for_counterparty(book, "CPTY_A", aaa, baa)  # mid
    spread_b = spread_for_counterparty(book, "CPTY_B", aaa, baa)  # smallest book
    assert spread_c < spread_a < spread_b


def test_build_credit_curves_one_per_counterparty(book):
    curves = build_credit_curves(book, REF_DATE)
    assert set(curves.keys()) == {"CPTY_A", "CPTY_B", "CPTY_C"}
    # CPTY_C (largest book) has the highest survival probability at any horizon
    assert curves["CPTY_C"].survival(date(2027, 1, 15)) > curves["CPTY_B"].survival(date(2027, 1, 15))


# ---------------------------------------------------------------- NEE + xVA, small simulation

@pytest.fixture(scope="module")
def small_run(book):
    trades = book
    factors = extract_factors(
        {k: v for k, v in trades.items() if k.startswith("EQTRS")},
        {k: v for k, v in trades.items() if k.startswith("BF")},
        {k: v for k, v in trades.items() if k.startswith("BTRS")},
    )
    max_maturity = max(getattr(t, "end_date", None) or t.forward_date for t in trades.values())
    grid = build_simulation_grid(REF_DATE, max_maturity)
    anchors = reporting_anchors(REF_DATE, max_maturity)
    regression_dates = collect_regression_dates(trades, anchors, ref_date=REF_DATE)

    usd_curve = zero_curve(REF_DATE, [0.5, 1, 2, 5, 10], [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])
    rate_calibrated = get_rate_model("LGM1F").calibrate(usd_curve, flat_vol_surface("RATE_USD", flat_vol=0.010))
    sim = JointSimulator()
    rf = factors.rates[0]
    sim.add_rate(rf, rate_calibrated)
    spot = {isin: 100.0 for isin in {e.isin for e in factors.equities}}
    div = {isin: 0.015 for isin in spot}
    for eq in factors.equities:
        c = get_spot_model("GBM").calibrate(spot[eq.isin], rate_calibrated, flat_vol_surface(str(eq), flat_vol=0.22), dividend_rate=div[eq.isin])
        sim.add_spot(eq, c, drift_rate_factor=rf)
    if factors.fx:
        fx_factor = factors.fx[0]
        implied_jpy = build_implied_jpy_curve(REF_DATE, 150.0, [(t, 150.0) for t in (0.5, 1, 2)], usd_curve)
        fxc = get_spot_model("FXGBM").calibrate(150.0, rate_calibrated, flat_vol_surface(str(fx_factor), flat_vol=0.10), implied_foreign_curve=implied_jpy)
        sim.add_spot(fx_factor, fxc, drift_rate_factor=rf)

    rng = np.random.default_rng(7)
    precache = sim.simulate(MarketState(ref_date=REF_DATE), n_paths=300, horizon_dates=grid.dates, rng=rng, ref_date=REF_DATE)
    result = price_curves(trades, precache, regression_dates, div, n_workers=1)
    counterparties = build_netting_hierarchy(trades)
    profiles = compute_all_profiles(counterparties, result, anchors, REF_DATE)

    credit_curves = build_credit_curves(trades, REF_DATE)
    own_credit = build_own_credit_curve(REF_DATE)
    per_cpty = {k: v for k, v in profiles.items() if k != "BOOK_TOTAL"}
    xva = compute_all_xva(per_cpty, credit_curves, own_credit, funding_spread=0.015,
                           discount_curve=usd_curve, ref_date=REF_DATE)
    return profiles, xva, usd_curve


def test_nee_always_nonpositive(small_run):
    profiles, _, _ = small_run
    for profile in profiles.values():
        assert all(n <= 1e-6 for n in profile.nee)


def test_cva_dva_fva_nonnegative(small_run):
    _, xva, _ = small_run
    for res in xva.values():
        assert res.cva >= 0.0
        assert res.dva >= 0.0
        assert res.fva >= 0.0


def test_cva_per_unit_ee_lower_for_better_credit_tier(small_run):
    """CPTY_C (largest book -> AAA-tier proxy, better assumed credit) should
    have a LOWER CVA per unit of EE than CPTY_B (smallest book -> BAA-tier,
    worse assumed credit) -- confirms the credit-tier weighting flows
    through the actual CVA integral, not just the raw spread numbers."""
    profiles, xva, _ = small_run
    ee_max_c = max(profiles["CPTY_C"].ee) if profiles["CPTY_C"].ee else 0.0
    ee_max_b = max(profiles["CPTY_B"].ee) if profiles["CPTY_B"].ee else 0.0
    if ee_max_c <= 0 or ee_max_b <= 0 or xva["CPTY_C"].cva <= 0 or xva["CPTY_B"].cva <= 0:
        pytest.skip("degenerate at this small path count -- EE or CVA is zero")
    ratio_c = xva["CPTY_C"].cva / ee_max_c
    ratio_b = xva["CPTY_B"].cva / ee_max_b
    assert ratio_c < ratio_b
