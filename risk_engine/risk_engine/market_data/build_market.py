"""
source_market_data: the orchestrator tying every module in market_data/
together into (1) a real capitolis_pricers.curves.Curve + FxCurve pair and
(2) a VolSurface per factor, ready to hand straight to the calibration layer
already built in risk_engine.models / risk_engine.calibration.

This is the actual "collect the market data" deliverable from MARKET_DATA.md
-- everywhere risk_engine's example scripts have used flat placeholder
numbers (spot=100, vol=0.22, correlations={}), this module fetches the real
(or best-available-free-proxy) equivalent instead. See each field's comment
below for exactly which prior placeholder it replaces and where the result
plugs into the existing pipeline.
"""
import csv
import os
from datetime import date

from capitolis_pricers.curves import zero_curve

from .rates import fetch_usd_curve_pillars
from .equity import fetch_all_equity_history
from .fx import fetch_fx_spot, fetch_fx_history, build_fx_forward_points
from .jpy_rate import flat_jpy_rate_by_tenor
from .vol_corr import realized_vol, realized_correlation_matrix
from ..calibration.market_surface import flat_vol_surface
from ..calibration.implied_fx_curve import build_implied_jpy_curve


def _load_isin_ticker_map(equities_csv_path: str):
    """isin -> ticker, read directly from equities.csv (the ticker column is
    dropped by capitolis_pricers.underlyings_loader.load_equities, which only
    keeps isin/shares/basis/currency -- ticker is Capitolis-provided mapping
    data, read here rather than re-deriving it)."""
    out = {}
    with open(equities_csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            isin = row["isin"].strip()
            ticker = row.get("ticker", "").strip()
            if isin and ticker:
                out[isin] = ticker
    return out


def source_market_data(equities_csv_path: str, ref_date: date, history_range: str = "1y",
                        include_fx: bool = True, vol_history_range: str = "1y"):
    """Source everything MARKET_DATA.md Sec.1-6 asks for, from free data
    where a free source exists (documented per-field below), and return it
    already packaged for the existing pipeline.

    Returns a dict:
        usd_curve            : capitolis_pricers.curves.Curve
                                -- replaces every example script's
                                   zero_curve(REF_DATE, [0.5,1,2,5,10], [flat numbers])
        equity_spot          : {isin: latest close}
                                -- replaces equity_spot = {isin: 100.0 for isin in ...}
        equity_vol_surface    : {isin: VolSurface}   (flat/ATM realized vol)
                                -- replaces flat_vol_surface(str(eq), flat_vol=0.22)
        fx_spot               : float (USDJPY) or None if include_fx=False
                                -- replaces fx_spot = 150.0
        implied_jpy_curve     : capitolis_pricers.curves.Curve or None
                                -- replaces build_implied_jpy_curve(..., [(t, fx_spot) ...])'s
                                   flat/no-differential placeholder pillars
        fx_vol_surface         : VolSurface or None
                                -- replaces flat_vol_surface(str(fx_factor), flat_vol=0.10)
        correlations           : {(factor_key_a, factor_key_b): float}
                                -- replaces the EMPTY correlations dict every prior run
                                   used (MarketState(ref_date=...) with no correlations
                                   kwarg at all -- see risk_engine/examples/exposure_profile.py),
                                   which meant every run so far was UNCORRELATED. Pass this
                                   straight into MarketState(correlations=...) for
                                   JointSimulator to actually use.
        equity_dividend_rates  : {isin: 0.0}
                                -- NOT sourced (see note below); zeroed, not faked non-zero.
    """
    isin_to_ticker = _load_isin_ticker_map(equities_csv_path)

    # --- USD curve: MARKET_DATA.md Sec.2.1 --------------------------------
    tenors, rates = fetch_usd_curve_pillars()
    usd_curve = zero_curve(ref_date, tenors, rates)

    # --- Equity spots + realized vol: MARKET_DATA.md Sec.3, Sec.5 ---------
    histories_by_ticker = fetch_all_equity_history(list(isin_to_ticker.values()), range_=history_range)
    equity_spot, equity_vol_surface = {}, {}
    for isin, ticker in isin_to_ticker.items():
        hist = histories_by_ticker.get(ticker)
        if hist is None:
            continue  # fetch failed for this name; caller decides how to backfill
        equity_spot[isin] = hist["close"][-1]
        vol = realized_vol(hist["close"])
        equity_vol_surface[isin] = flat_vol_surface(f"EQ_{isin}", flat_vol=vol)

    # --- Dividend rates: MARKET_DATA.md Sec.3 -- sourced via Databento -----
    # Yahoo's chart API (used above for spot/vol) does not carry a forward
    # dividend YIELD at all; MARKET_DATA.md explicitly leaves sourcing
    # method up to the student ("implied, put-call parity..."). Put-call
    # parity against REAL OPRA (US consolidated options tape) quotes,
    # sourced via Databento, is now used -- see market_data/
    # databento_dividends.py's module docstring for the exact method,
    # its American-vs-European limitation, and why this was NOT viable via
    # Yahoo (broken/zero bid-ask-OI fields, confirmed directly earlier in
    # this project). Names Databento cannot price (JPY-listed tickers --
    # OPRA only covers US-listed equity options -- or any name whose
    # extraction fails/returns no usable pair) fall back to 0.0, same as
    # before: an honest gap, not a faked non-zero number.
    from .databento_dividends import fetch_implied_dividend_yield
    equity_dividend_rates = {}
    for isin in equity_spot:
        ticker = isin_to_ticker.get(isin, "")
        q = None
        if ticker and "." not in ticker:   # crude US-listing filter: JPY tickers carry a numeric+suffix form (e.g. "6902.T"), never bare
            try:
                q = fetch_implied_dividend_yield(ticker, ref_date, equity_spot[isin], usd_curve)
            except Exception:
                q = None   # any fetch/parity failure for this one name -- fall back, don't abort the whole book
        equity_dividend_rates[isin] = q if q is not None else 0.0

    # --- FX: MARKET_DATA.md Sec.2.2 ----------------------------------------
    fx_spot = implied_jpy_curve = fx_vol_surface = fx_hist = None
    if include_fx:
        fx_spot = fetch_fx_spot()
        fx_hist = fetch_fx_history(range_=vol_history_range)
        fx_vol = realized_vol(fx_hist["close"])
        fx_vol_surface = flat_vol_surface("FX_USDJPY", flat_vol=fx_vol)

        usd_rate_by_tenor = dict(zip(tenors, rates))
        jpy_rate_by_tenor = flat_jpy_rate_by_tenor(tenors)  # single sourced JGB point, flat across tenors -- see jpy_rate.py
        forward_pillars = build_fx_forward_points(fx_spot, usd_rate_by_tenor, jpy_rate_by_tenor)
        implied_jpy_curve = build_implied_jpy_curve(ref_date, fx_spot, forward_pillars, usd_curve)

    # --- Correlations: MARKET_DATA.md Sec.6 --------------------------------
    price_histories = {f"EQ_{isin}": histories_by_ticker[isin_to_ticker[isin]]["close"]
                       for isin in equity_spot if isin_to_ticker[isin] in histories_by_ticker}
    if include_fx and fx_hist is not None:
        price_histories["FX_USDJPY"] = fx_hist["close"]
    correlations = realized_correlation_matrix(price_histories) if len(price_histories) >= 2 else {}

    return {
        "usd_curve": usd_curve,
        "equity_spot": equity_spot,
        "equity_vol_surface": equity_vol_surface,
        "equity_dividend_rates": equity_dividend_rates,
        "fx_spot": fx_spot,
        "implied_jpy_curve": implied_jpy_curve,
        "fx_vol_surface": fx_vol_surface,
        "correlations": correlations,
    }
