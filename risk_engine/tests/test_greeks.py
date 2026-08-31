"""
SA-CCR delta formula correctness, asset-class routing, and sign convention
tests -- all cheap (no simulation, closed-form), run against both small
synthetic trades and the real 16-trade book. Vega itself is verified only
by the end-to-end greeks_report.py smoke/full runs (documented in that
script's own module docstring as needing 10,000 paths for stable
95th/99th-percentile metrics), not by a fast unit test here.
"""
import math
import os
import sys
from datetime import date

import pytest

RISK_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICERS_ROOT = os.path.join(os.path.dirname(RISK_ENGINE_ROOT), "capitolis_pricers", "capitolis_pricers")
for p in (RISK_ENGINE_ROOT, PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.greeks.sa_ccr import (
    SUPERVISORY_FACTOR, supervisory_duration, rate_delta, fx_delta, equity_delta,
    trade_delta, aggregate_delta,
)
from risk_engine.greeks.sensitivities import asset_class_of, direction_sign, notional_of, is_compo


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


# ---------------------------------------------------------------- supervisory_duration

def test_supervisory_duration_zero_maturity_is_zero():
    assert supervisory_duration(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_supervisory_duration_known_value():
    # SD(0,1) = (e^0 - e^-0.05)/0.05
    expected = (1.0 - math.exp(-0.05)) / 0.05
    assert supervisory_duration(0.0, 1.0) == pytest.approx(expected)


def test_supervisory_duration_increases_with_maturity():
    """Longer-dated trades get a larger (but bounded, saturating) supervisory
    duration -- SD is monotonic increasing in E for fixed S."""
    sd_1y = supervisory_duration(0.0, 1.0)
    sd_5y = supervisory_duration(0.0, 5.0)
    sd_10y = supervisory_duration(0.0, 10.0)
    assert sd_1y < sd_5y < sd_10y
    assert sd_10y < 1.0 / 0.05  # bounded by the asymptote 1/alpha


# ---------------------------------------------------------------- asset class routing (real book)

def test_bond_trades_route_to_interest_rate(book):
    for tid, trade in book.items():
        if tid.startswith("BF") or tid.startswith("BTRS"):
            assert asset_class_of(trade) == "interest_rate", tid


def test_equity_trades_route_to_equity_or_compo(book):
    compo_ids = {"EQTRS_0005", "EQTRS_0006"}  # confirmed earlier this session: JPY-basket-row compo trades
    for tid, trade in book.items():
        if tid.startswith("EQTRS"):
            expected = "equity_fx_compo" if tid in compo_ids else "equity"
            assert asset_class_of(trade) == expected, tid


def test_trade_delta_compo_has_both_equity_and_fx_keys(book):
    for tid in ("EQTRS_0005", "EQTRS_0006"):
        deltas = trade_delta(book[tid], REF_DATE)
        assert set(deltas.keys()) == {"equity", "foreign_exchange"}


def test_trade_delta_plain_equity_has_only_equity_key(book):
    deltas = trade_delta(book["EQTRS_0001"], REF_DATE)
    assert set(deltas.keys()) == {"equity"}


def test_trade_delta_bond_forward_has_only_rate_key(book):
    deltas = trade_delta(book["BF_0001"], REF_DATE)
    assert set(deltas.keys()) == {"interest_rate"}


# ---------------------------------------------------------------- sign convention

def test_all_book_trades_are_short_capitolis_sells(book):
    """Confirmed by direct CSV inspection earlier this session: every trade
    in the shipped book is pay_equity/pay_tr/short -- Slide 6's "Capitolis
    is strictly seller on these transactions!" This test locks that fact in
    as a regression guard on direction_sign's convention, not just an
    incidental observation."""
    for trade in book.values():
        assert direction_sign(trade) == -1


def test_aggregate_delta_all_negative_matches_seller_convention(book):
    deltas = aggregate_delta(book, REF_DATE)
    for cpty, by_asset_class in deltas.items():
        for asset_class, delta in by_asset_class.items():
            assert delta < 0, f"{cpty}/{asset_class} delta should be negative (seller book)"


# ---------------------------------------------------------------- hand-calculation cross-check

def test_bf_0003_rate_delta_matches_hand_calculation(book):
    """BF_0003: notional 500,000,000, forward_date 2026-12-06, short.
    Cross-checked by hand during development; locking in as a regression test."""
    trade = book["BF_0003"]
    S, E = 0.0, (date(2026, 12, 6) - REF_DATE).days / 365.0
    expected = 500_000_000 * supervisory_duration(S, E) * -1 * SUPERVISORY_FACTOR["interest_rate"]
    assert rate_delta(trade, REF_DATE) == pytest.approx(expected)


# ---------------------------------------------------------------- aggregate_delta partitioning

def test_aggregate_delta_partitions_by_known_counterparties(book):
    deltas = aggregate_delta(book, REF_DATE)
    assert set(deltas.keys()) == {"CPTY_A", "CPTY_B", "CPTY_C"}
