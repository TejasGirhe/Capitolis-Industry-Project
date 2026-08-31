"""
Equity spot + historical daily prices -- sourced from Yahoo Finance's public
chart API, free, no key required (a browser-like User-Agent header is
needed; the bare requests default gets rate-limited/blocked).

equities.csv already carries both `ticker` (Yahoo-Finance-native format --
'6902.T' for Tokyo listings, 'BRK.B' for share classes -- confirmed by
inspection) and `isin` (the identifier the pricers actually key market data
by, per capitolis_pricers.market.MarketState.equity_spot(isin)). This module
fetches by ticker and lets the caller re-key the result by isin -- see
build_market.py for that mapping step; no separate ISIN resolver needed
since the ticker is already given in the trade data Capitolis provided.

Where this plugs in:
    MarketState.equity_spots[isin]          <- fetch_equity_history(...)['close'][-1]
    MarketState.equity_dividend_rates[isin] <- see build_market.py (dividend
        yield is a separate, harder-to-source quantity; not fetched here)
    risk_engine.calibration.market_surface  <- realized_vol() on the returns
        from this same price history (vol_corr.py), and
        realized_correlation_matrix() across multiple names' histories
"""
import requests

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_equity_history(ticker: str, range_: str = "1y", interval: str = "1d"):
    """Daily close price history for one ticker.

    Returns {'dates': [...], 'close': [...]} -- dates are UNIX timestamps
    (seconds), close are floats, both filtered to drop null bars (halts,
    pre-listing gaps).
    """
    params = {"range": range_, "interval": interval}
    resp = requests.get(YAHOO_CHART_URL.format(ticker=ticker), params=params, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise ValueError(f"no chart data for ticker '{ticker}': {error}")
    r = result[0]
    timestamps = r["timestamp"]
    closes = r["indicators"]["quote"][0]["close"]
    dates, out_closes = [], []
    for t, c in zip(timestamps, closes):
        if c is not None:
            dates.append(t)
            out_closes.append(float(c))
    if not out_closes:
        raise ValueError(f"ticker '{ticker}' returned no valid close prices")
    return {"dates": dates, "close": out_closes}


def fetch_all_equity_history(tickers, range_: str = "1y", interval: str = "1d", on_error="warn"):
    """{ticker: history_dict} for every ticker in `tickers`, skipping (and
    optionally warning on) any that fail -- a single bad/delisted ticker
    shouldn't abort sourcing the other 36+ names.

    on_error: 'warn' (print and skip) or 'raise' (propagate the first failure).
    """
    out = {}
    for ticker in tickers:
        try:
            out[ticker] = fetch_equity_history(ticker, range_, interval)
        except Exception as e:
            if on_error == "raise":
                raise
            print(f"[market_data] WARNING: failed to fetch '{ticker}': {e}")
    return out
