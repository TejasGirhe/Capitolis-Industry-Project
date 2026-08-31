"""
USDJPY FX spot + history -- sourced from Yahoo Finance ('USDJPY=X' ticker,
confirmed working). FX forward POINTS (the actual interbank-quoted product
MARKET_DATA.md Sec.2.2 asks for) are not free public data -- no free source
publishes real FX swap points. This module derives forward levels instead
via covered interest parity from the USD and (separately sourced) implied
JPY curve differential, which is the same approach
risk_engine.calibration.implied_fx_curve already uses in the other
direction (there: forward points -> implied JPY curve; here: rate
differential -> forward points, since we have no real points to invert).

Where this plugs in:
    fetch_fx_spot()              -> MarketState.fx_curves via FxCurve(...).spot,
                                     or directly as the `fx_spot` argument to
                                     FXGBM.calibrate(...)
    build_fx_forward_points(...) -> risk_engine.calibration.implied_fx_curve.
                                     build_implied_jpy_curve(...)'s
                                     `forward_pillars` argument
    fetch_fx_history()           -> realized_vol()/realized_correlation_matrix()
                                     for the FX_USDJPY factor's vol surface
"""
import math

import requests

from .equity import fetch_equity_history  # same Yahoo chart API, reused as-is

FX_TICKER = "USDJPY=X"


def fetch_fx_spot() -> float:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{FX_TICKER}",
        params={"range": "1d", "interval": "1d"}, headers=headers, timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"]
    if not result:
        raise ValueError("no FX spot data for USDJPY=X")
    return float(result[0]["meta"]["regularMarketPrice"])


def fetch_fx_history(range_: str = "1y", interval: str = "1d"):
    """Same shape as fetch_equity_history: {'dates': [...], 'close': [...]}."""
    return fetch_equity_history(FX_TICKER, range_, interval)


def build_fx_forward_points(fx_spot: float, usd_rate_by_tenor: dict, jpy_rate_by_tenor: dict):
    """Derive forward levels via covered interest parity: F(T) = S * (1 + r_usd*T) / (1 + r_jpy*T)
    (simple compounding proxy -- consistent with the free-data-only ethos of
    this module; a real desk would use the actual quoted swap points, not
    derive them, since forward points do NOT have to equal the CIP-implied
    level in practice -- basis exists. Documented approximation.)

    usd_rate_by_tenor / jpy_rate_by_tenor: {tenor_years: rate} dicts sharing
    the same tenor keys (see build_market.py for how the JPY side is sourced
    -- a JGB/Japan short-rate series, since there is no public JPY OIS feed
    either).

    Returns [(tenor_years, forward_level), ...] -- the `forward_pillars`
    shape risk_engine.calibration.implied_fx_curve.build_implied_jpy_curve expects.
    """
    pillars = []
    for tenor in sorted(set(usd_rate_by_tenor) & set(jpy_rate_by_tenor)):
        r_usd = usd_rate_by_tenor[tenor]
        r_jpy = jpy_rate_by_tenor[tenor]
        fwd = fx_spot * (1.0 + r_usd * tenor) / (1.0 + r_jpy * tenor)
        pillars.append((tenor, fwd))
    if not pillars:
        raise ValueError("no overlapping tenors between usd_rate_by_tenor and jpy_rate_by_tenor")
    return pillars
