"""
Regression guard for factor extraction against the real trade_data/ book.

Ground truth (verified by direct inspection of equity_trs.csv / equities.csv):
16 trades total; 41 basket rows collapse to 37 unique ISINs (4 names repeat
across trades); EQTRS_0005 and EQTRS_0006 hold JPY-currency basket rows
against USD trade_ccy -- i.e. two compo trades -- so exactly one FxFactor
(USD, JPY) is expected, not zero. This corrects an earlier assumption (based
on the kickoff deck's "1 JPY compo trade" claim) that the book had no compo
exposure; extraction must be verified against data, not the deck.
"""
import os
import sys

import pytest

RISK_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICERS_ROOT = os.path.join(os.path.dirname(RISK_ENGINE_ROOT), "capitolis_pricers", "capitolis_pricers")
for p in (RISK_ENGINE_ROOT, PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

from risk_engine.factors.extract import extract_factors
from risk_engine.factors.types import RateFactor, EquityFactor, FxFactor, CreditFactor


TRADE_DATA = os.path.join(PRICERS_ROOT, "trade_data")


@pytest.fixture(scope="module")
def loaded_book():
    baskets = load_equities(os.path.join(TRADE_DATA, "underlyings", "equities.csv"))
    bonds = load_bonds(os.path.join(TRADE_DATA, "underlyings", "bonds.csv"))
    eqtrs = load_equity_trs(os.path.join(TRADE_DATA, "equity_trs.csv"), baskets)
    bfwd = load_bond_forward(os.path.join(TRADE_DATA, "bond_forward.csv"), bonds)
    btrs = load_bond_trs(os.path.join(TRADE_DATA, "bond_trs.csv"), bonds)
    return eqtrs, bfwd, btrs, bonds


def test_rate_factor_is_usd_only(loaded_book):
    eqtrs, bfwd, btrs, _ = loaded_book
    fs = extract_factors(eqtrs, bfwd, btrs)
    assert fs.rates == (RateFactor("USD"),)


def test_equity_factor_count_matches_unique_isins(loaded_book):
    eqtrs, bfwd, btrs, _ = loaded_book
    fs = extract_factors(eqtrs, bfwd, btrs)
    assert len(fs.equities) == 37
    assert all(isinstance(e, EquityFactor) for e in fs.equities)


def test_fx_factor_detected_for_compo_trades(loaded_book):
    eqtrs, bfwd, btrs, _ = loaded_book
    fs = extract_factors(eqtrs, bfwd, btrs)
    assert fs.fx == (FxFactor("USD", "JPY"),)


def test_no_credit_factors_by_default(loaded_book):
    eqtrs, bfwd, btrs, _ = loaded_book
    fs = extract_factors(eqtrs, bfwd, btrs)
    assert fs.credit == ()


def test_credit_factors_when_risky_bond_extension_enabled(loaded_book):
    eqtrs, bfwd, btrs, bonds = loaded_book
    fs = extract_factors(eqtrs, bfwd, btrs, include_risky_bond_credit=True)
    # bonds.csv populates issuer='US TREASURY N/B' for every bond -- so the
    # optional extension, if switched on, would treat the book's Treasuries
    # as credit-risky. This is almost certainly not intended (Treasuries are
    # the canonical risk-free asset and README Sec.7/8 describes them as
    # such) but it is what the data says, so the flag is off by default and
    # this test documents the behavior rather than silently assuming blank.
    assert fs.credit == (CreditFactor("US TREASURY N/B"),)


def test_total_factor_count(loaded_book):
    eqtrs, bfwd, btrs, _ = loaded_book
    fs = extract_factors(eqtrs, bfwd, btrs)
    assert len(fs) == 1 + 37 + 1 + 0


def test_empty_book_returns_empty_factor_set():
    fs = extract_factors()
    assert len(fs) == 0
