"""
Per-trade independent simulation mode: factor-subset correctness (each
trade's own Cholesky covers only its own factors), genuine independence
(no accidental shared draws across trades), per-trade exposure sanity, and
that the independent aggregate is visibly, meaningfully different from the
joint-mode (correlated) counterparty exposure -- confirming the
independence assumption actually takes effect rather than silently
degenerating to the joint numbers.
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

from capitolis_pricers.curves import zero_curve
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.simulation.per_trade import (
    extract_trade_factors, build_trade_simulator, simulate_trade, simulate_all_trades_independently,
)
from risk_engine.simulation.grid import build_simulation_grid, reporting_anchors, collect_regression_dates
from risk_engine.pricing import price_curves, price_curves_per_trade
from risk_engine.exposure import compute_all_profiles, compute_per_trade_profiles
from risk_engine.netting import build_netting_hierarchy, aggregate_all_independent_profiles


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


@pytest.fixture(scope="module")
def usd_curve():
    return zero_curve(REF_DATE, [0.5, 1, 2, 5, 10], [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])


# ---------------------------------------------------------------- factor-subset correctness

def test_plain_equity_trs_has_rate_and_equity_factors_only(book):
    factors = extract_trade_factors(book["EQTRS_0001"])
    assert len(factors.rates) == 1
    assert len(factors.equities) >= 1
    assert len(factors.fx) == 0


def test_compo_equity_trs_has_fx_factor_too(book):
    factors = extract_trade_factors(book["EQTRS_0005"])
    assert len(factors.rates) == 1
    assert len(factors.equities) >= 1
    assert len(factors.fx) == 1


def test_bond_forward_has_rate_factor_only(book):
    factors = extract_trade_factors(book["BF_0001"])
    assert len(factors.rates) == 1
    assert len(factors.equities) == 0
    assert len(factors.fx) == 0


def test_trade_simulator_cholesky_scoped_to_trade_factors(book, usd_curve):
    """The Cholesky/correlation matrix size is driven by how many factors
    were add_rate/add_spot'd -- confirm a plain equity TRS's simulator has
    exactly (1 rate + n_equity_names) factors added, not the whole book's 39."""
    trade = book["EQTRS_0001"]
    isins = [p.isin for p in trade.positions]
    spot = {i: 100.0 for i in isins}
    vol_surface = {i: flat_vol_surface(f"EQ_{i}", flat_vol=0.22) for i in isins}
    div = {i: 0.015 for i in isins}
    sim = build_trade_simulator(trade, "LGM1F", "GBM", "FXGBM", usd_curve,
                                 flat_vol_surface("RATE_USD", flat_vol=0.01), spot, vol_surface, div)
    assert len(sim._rate_factors) == 1
    assert len(sim._spot_factors) == len(isins)


# ---------------------------------------------------------------- genuine independence

def test_two_trades_sharing_rate_factor_get_different_paths(book, usd_curve):
    """EQTRS_0001 and BF_0001 both reference RateFactor(USD) (equal by
    value) but must NOT receive numerically identical simulated paths --
    confirms real independence, not an accidental shared-draw leak."""
    eq_trade, bond_trade = book["EQTRS_0001"], book["BF_0001"]
    rate_vol = flat_vol_surface("RATE_USD", flat_vol=0.01)
    isins = [p.isin for p in eq_trade.positions]
    spot = {i: 100.0 for i in isins}
    vol_surface = {i: flat_vol_surface(f"EQ_{i}", flat_vol=0.22) for i in isins}
    div = {i: 0.015 for i in isins}

    sim1 = build_trade_simulator(eq_trade, "LGM2F_SV", "GBM_SV", "FXGBM_SV", usd_curve, rate_vol, spot, vol_surface, div)
    sim2 = build_trade_simulator(bond_trade, "LGM2F_SV", "GBM_SV", "FXGBM_SV", usd_curve, rate_vol, {}, {}, {})

    horizon = [REF_DATE + timedelta(days=30)]
    r1 = simulate_trade("EQTRS_0001", eq_trade, sim1, 100, horizon, REF_DATE)
    r2 = simulate_trade("BF_0001", bond_trade, sim2, 100, horizon, REF_DATE)

    rf1 = next(iter(r1.rate_states))
    rf2 = next(iter(r2.rate_states))
    assert rf1 == rf2  # same factor identity
    assert not np.allclose(r1.rate_states[rf1], r2.rate_states[rf2])  # but different simulated values


def test_same_trade_id_reproducible_seed(book, usd_curve):
    """Same trade_id -> same seed -> same simulated path, for reproducibility."""
    trade = book["BF_0001"]
    rate_vol = flat_vol_surface("RATE_USD", flat_vol=0.01)
    horizon = [REF_DATE + timedelta(days=30)]

    sim_a = build_trade_simulator(trade, "LGM1F", "GBM", "FXGBM", usd_curve, rate_vol, {}, {}, {})
    sim_b = build_trade_simulator(trade, "LGM1F", "GBM", "FXGBM", usd_curve, rate_vol, {}, {}, {})
    r_a = simulate_trade("BF_0001", trade, sim_a, 50, horizon, REF_DATE)
    r_b = simulate_trade("BF_0001", trade, sim_b, 50, horizon, REF_DATE)

    rf = next(iter(r_a.rate_states))
    assert np.array_equal(r_a.rate_states[rf], r_b.rate_states[rf])


# ---------------------------------------------------------------- per-trade exposure sanity + aggregate divergence

@pytest.fixture(scope="module")
def cpty_a_trades(book):
    return {tid: t for tid, t in book.items() if t.counterparty == "CPTY_A"}


@pytest.fixture(scope="module")
def per_trade_profiles(cpty_a_trades, usd_curve):
    trades = cpty_a_trades
    max_maturity = max(getattr(t, "end_date", None) or t.forward_date for t in trades.values())
    grid = build_simulation_grid(REF_DATE, max_maturity)
    anchors = reporting_anchors(REF_DATE, max_maturity)
    regression_dates = collect_regression_dates(trades, anchors, ref_date=REF_DATE)

    rate_vol = flat_vol_surface("RATE_USD", flat_vol=0.01)
    all_isins = {p.isin for t in trades.values() if hasattr(t, "positions") for p in t.positions}
    spot = {i: 100.0 for i in all_isins}
    vol_surface = {i: flat_vol_surface(f"EQ_{i}", flat_vol=0.22) for i in all_isins}
    div = {i: 0.015 for i in all_isins}

    precache = simulate_all_trades_independently(
        trades, 300, grid.dates, REF_DATE, "LGM1F", "GBM", "FXGBM", usd_curve, rate_vol, spot, vol_surface, div)
    curves = price_curves_per_trade(trades, precache, regression_dates, div, n_workers=1)
    return compute_per_trade_profiles(trades, curves, anchors, REF_DATE)


def test_per_trade_ee_nonnegative(per_trade_profiles):
    for profile in per_trade_profiles.values():
        assert all(e >= -1e-6 for e in profile.ee)


def test_per_trade_pfe_at_least_ee_when_ee_positive(per_trade_profiles):
    """PFE_99 >= EE only holds when EE > 0 (EE floors at 0 via max(x,0), but
    PFE_99 is an unfloored quantile -- if a trade's exposure is negative on
    essentially every path, EE=0.0 exactly while PFE_99 can legitimately be
    negative too. Confirmed not a bug: np.quantile of an all-negative
    sample is itself negative, and mean(max(x,0)) over the same sample is
    exactly 0 -- both correct given the same input, just not comparable by
    a blanket pfe>=ee when ee==0."""
    for profile in per_trade_profiles.values():
        for ee_t, pfe_t in zip(profile.ee, profile.pfe_99):
            if ee_t > 1e-9:
                assert pfe_t >= ee_t - 1e-6


def test_per_trade_mpe_equals_max_pfe(per_trade_profiles):
    for profile in per_trade_profiles.values():
        assert profile.mpe_99 == pytest.approx(max(profile.pfe_99)) if profile.pfe_99 else profile.mpe_99 == 0.0


def test_independent_aggregate_diverges_from_joint_netted_exposure(cpty_a_trades, per_trade_profiles, usd_curve):
    """The independent-mode aggregate must NOT equal the joint-mode
    (correlated) netted exposure for the same counterparty/book -- a
    regression guard that per-trade mode's independence assumption is
    actually taking effect, not silently collapsing to the joint numbers."""
    from risk_engine.simulation.joint import JointSimulator
    from risk_engine.models.registry import get_rate_model, get_spot_model

    trades = cpty_a_trades
    max_maturity = max(getattr(t, "end_date", None) or t.forward_date for t in trades.values())
    grid = build_simulation_grid(REF_DATE, max_maturity)
    anchors = reporting_anchors(REF_DATE, max_maturity)
    regression_dates = collect_regression_dates(trades, anchors, ref_date=REF_DATE)

    rate_vol = flat_vol_surface("RATE_USD", flat_vol=0.01)
    all_isins = {p.isin for t in trades.values() if hasattr(t, "positions") for p in t.positions}
    spot = {i: 100.0 for i in all_isins}
    div = {i: 0.015 for i in all_isins}

    rate_calibrated = get_rate_model("LGM1F").calibrate(usd_curve, rate_vol)
    sim = JointSimulator()
    from risk_engine.factors.extract import extract_factors
    eq_only = {k: v for k, v in trades.items() if k.startswith("EQTRS")}
    bf_only = {k: v for k, v in trades.items() if k.startswith("BF")}
    bt_only = {k: v for k, v in trades.items() if k.startswith("BTRS")}
    factors = extract_factors(eq_only, bf_only, bt_only)
    rate_factor = factors.rates[0]
    sim.add_rate(rate_factor, rate_calibrated)
    for eq_factor in factors.equities:
        calibrated = get_spot_model("GBM").calibrate(
            spot[eq_factor.isin], rate_calibrated, flat_vol_surface(f"EQ_{eq_factor.isin}", flat_vol=0.22),
            dividend_rate=div[eq_factor.isin])
        sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)

    from capitolis_pricers.market import MarketState
    rng = np.random.default_rng(42)
    joint_precache = sim.simulate(MarketState(ref_date=REF_DATE), n_paths=300, horizon_dates=grid.dates, rng=rng, ref_date=REF_DATE)
    joint_result = price_curves(trades, joint_precache, regression_dates, div, n_workers=1)
    counterparties = build_netting_hierarchy(trades)
    joint_profiles = compute_all_profiles(counterparties, joint_result, anchors, REF_DATE)
    joint_cpty_a = joint_profiles["CPTY_A"]

    independent_agg = aggregate_all_independent_profiles(trades, per_trade_profiles)["CPTY_A"]

    # both nonzero and not coincidentally equal (different simulation
    # mechanisms, different rng streams -- should differ meaningfully)
    assert joint_cpty_a.mpe_99 > 0
    assert independent_agg.mpe_aggregate_99 > 0
    assert joint_cpty_a.mpe_99 != pytest.approx(independent_agg.mpe_aggregate_99, rel=0.01)
