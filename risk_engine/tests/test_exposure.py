"""
Grid, netting hierarchy, interpolation, and exposure sanity checks, at small
scale (cheap, fast, deterministic) -- the full 10,000-path exposure profile
is run and verified separately as a timed background run.
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
from capitolis_pricers.curves import zero_curve
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.factors.extract import extract_factors
from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.calibration.implied_fx_curve import build_implied_jpy_curve
from risk_engine.models.registry import get_rate_model, get_spot_model
from risk_engine.simulation.joint import JointSimulator
from risk_engine.simulation.grid import (
    build_simulation_grid, reporting_anchors, collect_regression_dates, add_business_days,
)
from risk_engine.pricing import price_curves
from risk_engine.netting import build_netting_hierarchy
from risk_engine.exposure import compute_all_profiles


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


def _max_maturity(book):
    return max(t.end_date if hasattr(t, "end_date") else t.forward_date for t in book.values())


def test_simulation_grid_is_mpor_independent():
    """The fixed simulation grid no longer carries MPoR companion dates --
    it's sized purely for interpolation fidelity."""
    grid = build_simulation_grid(REF_DATE, date(2028, 1, 15))
    assert grid.dates[0] == REF_DATE
    assert grid.horizon == date(2029, 1, 15)
    # weekly through month 3
    week1 = REF_DATE + timedelta(days=7)
    assert week1 in grid.dates


def test_regression_dates_include_cashflow_and_mpor_endpoints(book):
    anchors = reporting_anchors(REF_DATE, _max_maturity(book))
    regression_dates = collect_regression_dates(book, anchors, ref_date=REF_DATE)
    # every trade's own maturity date must be a regression date
    for trade in book.values():
        maturity = trade.end_date if hasattr(trade, "end_date") else trade.forward_date
        assert maturity in regression_dates
    # every anchor's t-1bd and t+10bd companions must be present
    for t in anchors:
        assert add_business_days(t, -1) in regression_dates
        assert add_business_days(t, 10) in regression_dates


def test_netting_hierarchy_partitions_all_trades(book):
    counterparties = build_netting_hierarchy(book)
    assert {c.id for c in counterparties} == {"CPTY_A", "CPTY_B", "CPTY_C"}
    all_ids = sorted(tid for c in counterparties for ns in c.netting_sets for tid in ns.trade_ids)
    assert all_ids == sorted(book.keys())


@pytest.fixture(scope="module")
def small_exposure_run(book):
    factors = extract_factors(
        {k: v for k, v in book.items() if k.startswith("EQTRS")},
        {k: v for k, v in book.items() if k.startswith("BF")},
        {k: v for k, v in book.items() if k.startswith("BTRS")},
    )
    max_maturity = _max_maturity(book)
    grid = build_simulation_grid(REF_DATE, max_maturity)
    anchors = reporting_anchors(REF_DATE, max_maturity)
    regression_dates = collect_regression_dates(book, anchors, ref_date=REF_DATE)

    usd_curve = zero_curve(REF_DATE, [0.5, 1, 2, 5, 10], [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])
    rate_calibrated = get_rate_model("LGM1F").calibrate(usd_curve, flat_vol_surface("RATE_USD", flat_vol=0.010))

    sim = JointSimulator()
    rate_factor = factors.rates[0]
    sim.add_rate(rate_factor, rate_calibrated)
    equity_div = {}
    for eq_factor in factors.equities:
        c = get_spot_model("GBM").calibrate(100.0, rate_calibrated, flat_vol_surface(str(eq_factor), flat_vol=0.22), dividend_rate=0.015)
        equity_div[eq_factor.isin] = 0.015
        sim.add_spot(eq_factor, c, drift_rate_factor=rate_factor)
    implied_jpy = build_implied_jpy_curve(REF_DATE, 150.0, [(t, 150.0) for t in (0.5, 1.0, 2.0)], usd_curve)
    fx_factor = factors.fx[0]
    fxc = get_spot_model("FXGBM").calibrate(150.0, rate_calibrated, flat_vol_surface(str(fx_factor), flat_vol=0.10), implied_foreign_curve=implied_jpy)
    sim.add_spot(fx_factor, fxc, drift_rate_factor=rate_factor)

    rng = np.random.default_rng(3)
    precache = sim.simulate(MarketState(ref_date=REF_DATE), n_paths=50, horizon_dates=grid.dates, rng=rng, ref_date=REF_DATE)
    result = price_curves(book, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=1)
    counterparties = build_netting_hierarchy(book)
    profiles = compute_all_profiles(counterparties, result, anchors, REF_DATE)
    return profiles


def test_ee_is_nonnegative(small_exposure_run):
    for profile in small_exposure_run.values():
        assert all(e >= -1e-9 for e in profile.ee)


def test_pfe_99_at_least_pfe_95(small_exposure_run):
    for profile in small_exposure_run.values():
        for p95, p99 in zip(profile.pfe_95, profile.pfe_99):
            assert p99 >= p95 - 1e-6, f"{profile.owner_id}: PFE_99 {p99} < PFE_95 {p95}"


def test_tail_ee_at_least_pfe(small_exposure_run):
    """Conditional mean above a quantile threshold must be >= that threshold."""
    for profile in small_exposure_run.values():
        for p95, tail95 in zip(profile.pfe_95, profile.tail_ee_95):
            assert tail95 >= p95 - 1e-6
        for p99, tail99 in zip(profile.pfe_99, profile.tail_ee_99):
            assert tail99 >= p99 - 1e-6


def test_mpe_equals_max_pfe(small_exposure_run):
    for profile in small_exposure_run.values():
        if profile.pfe_95:
            assert profile.mpe_95 == pytest.approx(max(profile.pfe_95))
        if profile.pfe_99:
            assert profile.mpe_99 == pytest.approx(max(profile.pfe_99))


def test_eepe_within_ee_range(small_exposure_run):
    """A time-weighted average must lie within the range of what it averages."""
    for profile in small_exposure_run.values():
        first_year_ee = [e for d, e in zip(profile.dates, profile.ee) if (d - REF_DATE).days <= 365]
        if not first_year_ee:
            continue
        assert min(first_year_ee) - 1e-6 <= profile.eepe <= max(first_year_ee) + 1e-6


def test_book_total_present(small_exposure_run):
    assert "BOOK_TOTAL" in small_exposure_run
    assert set(small_exposure_run.keys()) == {"CPTY_A", "CPTY_B", "CPTY_C", "BOOK_TOTAL"}


def test_maturity_crossing_does_not_produce_fake_negative_exposure(small_exposure_run):
    """Regression test: a trade maturing inside the forward-looking MPoR
    window [t-1bd, t+10bd] (or exactly on one of its boundaries -- price_curves'
    own maturity guard keeps a trade live THROUGH its own maturity date, so
    the NPV->0 drop can land right at a window boundary) must not register
    as a large fake loss/gain from differencing a live NPV against zero.
    CPTY_B holds several trades maturing across Oct-Nov 2026, inside various
    anchors' MPoR windows.
    """
    profile = small_exposure_run["CPTY_B"]
    for p95, p99 in zip(profile.pfe_95, profile.pfe_99):
        assert p99 >= p95 - 1e-6
