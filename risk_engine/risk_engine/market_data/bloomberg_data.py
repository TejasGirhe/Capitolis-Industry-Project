"""
Loaders for the real Bloomberg-terminal data drop (data_bloomberg.zip,
manifest at data_bloomberg/metadata/manifest.csv) -- REPLACES several
proxies this project used earlier in its life wherever a genuine
substitute now exists:

  - USD discount curve: was FRED-pillar bootstrap (market_data/rates.py) ->
    now the Bloomberg SWPM zero-curve export (already has DF's computed by
    Bloomberg, no bootstrapping needed). FRED remains a documented fallback
    if this file is ever missing.
  - USD rate-vol surface: was the Databento SOFR-futures-options Bachelier
    proxy (market_data/databento_rates.py, built earlier this session) ->
    now the REAL USD SOFR ATM normal swaption cube (the actual OTC
    instrument, not a listed-futures proxy).
  - JPY discount curve: was a flat placeholder (market_data/jpy_rate.py) ->
    now a real Bloomberg JPY OIS SWPM zero curve.
  - JPY rate-vol surface: had NO source at all before -> now the real JPY
    OIS ATM normal swaption cube.
  - USDJPY spot/forward history and implied-vol surface: was Yahoo spot +
    realized vol -> now real Bloomberg BGN spot/forward/vol history.
  - SPX/TOPIX price, dividend yield, and implied vol surface: real
    Bloomberg index-level data, used (per the confirmed design) to derive a
    market-level implied/realized vol RATIO applied to the book's 37
    single-name realized vols -- see equity_vol_scaling_factor below.

This module only PARSES the CSVs and returns plain data structures
(Curve, VolSurface, DataFrames/dicts of history) -- it does not decide
which of these to use where; that wiring lives in
risk_engine/examples/_sourced_book.py.
"""
import csv
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from capitolis_pricers.curves import Curve
from ..calibration.market_surface import VolSurface, tenor_to_years

DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_bloomberg"))


def _path(*parts):
    return os.path.join(DATA_ROOT, *parts)


def is_available() -> bool:
    return os.path.isdir(DATA_ROOT)


# ================================================================ CURVES (SWPM zero curves)
def load_zero_curve(csv_path: str, ref_date: date) -> Curve:
    """Bloomberg SWPM zero-curve export -> capitolis_pricers.curves.Curve,
    built DIRECTLY from the tenor/discount_factor columns Bloomberg already
    computed (no bootstrapping) -- columns confirmed directly against the
    real file: valuation_date, curve, side, interpolation, tenor,
    market_rate_pct, zero_rate_pct, discount_factor, source_image."""
    tenors, dfs = [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            t = tenor_to_years(row["tenor"].strip())
            df = float(row["discount_factor"])
            tenors.append(t)
            dfs.append(df)
    return Curve(ref_date, tenors, dfs)


def load_usd_curve(ref_date: date) -> Curve:
    return load_zero_curve(_path("rates", "usd_sofr_bloomberg_zero_discount_curve_2026-08-31.csv"), ref_date)


def load_jpy_curve(ref_date: date) -> Curve:
    return load_zero_curve(_path("rates", "jpy_ois_bloomberg_zero_discount_curve_2026-08-31.csv"), ref_date)


# ================================================================ SWAPTION VOL CUBES -> VolSurface
def load_swaption_vol_surface(csv_path: str, factor_key: str) -> VolSurface:
    """VCUB ATM normal swaption cube (wide: rows=expiry, cols=swap tenor,
    values=normal vol in bp) -> a tenor x strike VolSurface, where "strike"
    here is repurposed as the SWAP TENOR axis (ATM only -- no real strike/
    moneyness axis exists in an ATM-only cube). This still gives
    models._shared.fit_skew_smile something to work with (>=3 points along
    that second axis at the longest expiry), on the honest basis that the
    axis represents swap-tenor structure, not moneyness -- documented here,
    not silently mislabeled. Vol values are converted from bp (e.g. 80.3)
    to decimal-normal-vol units (0.00803) for consistency with this
    project's other vol inputs, which are all decimal.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    swap_tenors = [tenor_to_years(h.strip()) for h in header[1:]]
    tenors, vols = [], {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        expiry = tenor_to_years(row[0].strip())
        tenors.append(expiry)
        for swap_tenor, cell in zip(swap_tenors, row[1:]):
            if cell.strip() == "":
                continue
            vols[(expiry, swap_tenor)] = float(cell) / 10000.0   # bp -> decimal
    return VolSurface(factor_key=factor_key, tenors=sorted(set(tenors)),
                      strikes=sorted(set(swap_tenors)), vols=vols)


def load_usd_swaption_vol_surface() -> VolSurface:
    return load_swaption_vol_surface(
        _path("rates", "usd_sofr_atm_normal_swaption_vol_2026-08-31_wide.csv"), "RATE_USD")


def load_jpy_swaption_vol_surface() -> VolSurface:
    return load_swaption_vol_surface(
        _path("rates", "jpy_ois_atm_normal_swaption_vol_2026-08-31_wide.csv"), "RATE_JPY")


# ================================================================ USDJPY SPOT / FORWARD / VOL
def load_usdjpy_spot_history() -> List[Tuple[date, float]]:
    """[(date, mid_spot), ...] sorted by date, from the real Bloomberg
    USDJPY spot history (px_last / bid / ask / mid columns; mid used)."""
    out = []
    with open(_path("fx", "usdjpy_spot_2017-01-01_2026-08-31.csv"), newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            d = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date()
            mid = row.get("mid_from_bid_ask") or row.get("px_last")
            if mid:
                out.append((d, float(mid)))
    return sorted(out)


def load_usdjpy_latest_spot() -> float:
    hist = load_usdjpy_spot_history()
    return hist[-1][1]


def load_usdjpy_implied_vol_history(component: str = "ATM", tenor: str = "3M") -> List[Tuple[date, float]]:
    """[(date, vol_pct)] for one (component, tenor) slice of the real
    historical USDJPY implied-vol surface (long format: date, component,
    tenor, ticker, value). component in {ATM, 25RR, 25BF, 10RR, 10BF}."""
    out = []
    path = _path("fx", "usdjpy_implied_vol_surface_long_2017-01-01_2026-08-31.csv")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["component"].strip() != component or row["tenor"].strip() != tenor:
                continue
            d = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date()
            v = row["value"].strip()
            if v:
                out.append((d, float(v)))
    return sorted(out)


def load_usdjpy_vol_surface_snapshot() -> VolSurface:
    """Real USDJPY ATM/RR/BF bid-ask snapshot (2026-08-31) -> a VolSurface
    keyed purely on ATM vol by expiry (RR/BF give the smile shape but this
    project's VolSurface strike axis expects a plain strike/moneyness grid,
    not a delta-quoted RR/BF pair -- using ATM-only here is honest: it is a
    real term structure, not a fabricated smile from RR/BF without doing
    the delta-to-strike conversion properly, which is out of scope for
    this pass)."""
    tenors, vols = [], {}
    path = _path("fx", "usdjpy_vol_surface_bid_ask_2026-08-31.csv")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            t = tenor_to_years(row["expiry"].strip())
            atm_mid = (float(row["atm_bid"]) + float(row["atm_ask"])) / 2.0 / 100.0   # pct -> decimal
            tenors.append(t)
            vols[(t, 1.0)] = atm_mid
    return VolSurface(factor_key="FX_USDJPY", tenors=sorted(set(tenors)), strikes=[1.0], vols=vols)


# ================================================================ EQUITY (SPX/TOPIX)
def load_index_history(csv_path: str) -> List[Dict]:
    """[{date, px_last, total_return_index, dividend_yield_12m}, ...] sorted
    by date, from the real SPX/TOPIX Bloomberg history files."""
    out = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            d = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date()
            out.append({
                "date": d,
                "px_last": float(row["px_last"]) if row["px_last"].strip() else None,
                "total_return_index": (float(row["total_return_index_gross_dvds"])
                                        if row["total_return_index_gross_dvds"].strip() else None),
                "dividend_yield_12m": (float(row["dividend_yield_12m_pct"]) / 100.0
                                        if row["dividend_yield_12m_pct"].strip() else None),
            })
    return sorted(out, key=lambda r: r["date"])


def load_spx_history() -> List[Dict]:
    return load_index_history(_path("equity", "spx_equity_history_2017-01-01_2026-08-31.csv"))


def load_topix_history() -> List[Dict]:
    return load_index_history(_path("equity", "topix_equity_history_2017-01-01_2026-08-31.csv"))


def load_index_vol_surface(csv_path: str, factor_key: str) -> VolSurface:
    """OVDV tenor x moneyness surface (wide) -> VolSurface -- real columns
    confirmed directly: tenor, expiry_date, implied_forward,
    iv_{80,90,95,97.5,100,102.5,105,110,120}pct_mny."""
    moneyness_cols = ["80", "90", "95", "97.5", "100", "102.5", "105", "110", "120"]
    tenors, strikes, vols = [], set(), {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            t = tenor_to_years(row["tenor"].strip())
            tenors.append(t)
            for mc in moneyness_cols:
                col = f"iv_{mc}pct_mny"
                if col not in row or not row[col].strip():
                    continue
                moneyness = float(mc) / 100.0
                strikes.add(moneyness)
                vols[(t, moneyness)] = float(row[col]) / 100.0   # pct -> decimal
    return VolSurface(factor_key=factor_key, tenors=sorted(set(tenors)), strikes=sorted(strikes), vols=vols)


def load_spx_vol_surface() -> VolSurface:
    return load_index_vol_surface(_path("equity", "spx_vol_surface_2026-08-31_wide.csv"), "SPX")


def load_topix_vol_surface() -> VolSurface:
    return load_index_vol_surface(_path("equity", "topix_vol_surface_2026-08-31_wide.csv"), "TOPIX")


def index_realized_vol(history: List[Dict], lookback_days: int = 252) -> float:
    """Annualized realized vol of an index's own price history, over the
    trailing `lookback_days` -- used as the denominator of the implied/
    realized ratio in equity_vol_scaling_factor below. Reuses the same
    method (log-return population stdev x sqrt(252)) as
    market_data.vol_corr.realized_vol, applied here to px_last rather than
    re-importing that module's Yahoo-shaped input contract."""
    import math
    closes = [r["px_last"] for r in history[-(lookback_days + 1):] if r["px_last"] is not None]
    if len(closes) < 2:
        raise ValueError("not enough index history to compute realized vol")
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def equity_vol_scaling_factor(index: str, atm_tenor_years: float = 0.25) -> float:
    """The market-level implied/realized vol RATIO for 'SPX' or 'TOPIX',
    at approximately `atm_tenor_years` (default ~3M, matching this
    project's own equity realized-vol lookback window) -- e.g. a ratio of
    1.15 means the real options market is pricing that index's vol 15%
    above its own trailing realized vol right now, which is a real,
    market-observed implied-risk-premium signal.

    Used by _sourced_book.py to scale EVERY single-name realized vol in
    that index's currency bucket (US-listed names by SPX, JPY-listed names
    by TOPIX) uniformly -- an explicit, confirmed design choice (a market-
    level adjustment, not a per-name beta regression) -- rather than to
    replace single-name vol outright (no valid single-name substitute
    exists in this data drop; using the RAW index vol AS a single-name
    vol would be a worse proxy than realized vol, not an upgrade).
    """
    if index == "SPX":
        history = load_spx_history()
        vol_surface = load_spx_vol_surface()
    elif index == "TOPIX":
        history = load_topix_history()
        vol_surface = load_topix_vol_surface()
    else:
        raise ValueError(f"index must be 'SPX' or 'TOPIX', got {index!r}")

    realized = index_realized_vol(history)
    # nearest available ATM-ish tenor's vol at the surface's own ATM moneyness node (1.0 / 100%)
    atm_tenor = min(vol_surface.tenors, key=lambda t: abs(t - atm_tenor_years))
    implied = vol_surface.vol(atm_tenor, 1.0)
    return implied / realized
