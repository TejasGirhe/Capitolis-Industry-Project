"""
JPY rate proxy -- for the FX forward covered-interest-parity derivation in
fx.py only (there is no JPY rate FACTOR in this book; MARKET_DATA.md
Sec.2.2 is explicit that no JPY curve is built, the rate is only ever
embedded in FX forwards).

Honest limitation: FRED does not carry a JPY short-rate/OIS term structure
comparable to the USD SOFR+Treasury ladder in rates.py -- only a single
long-tenor series (Japan 10Y government bond yield, monthly). Rather than
force a fabricated multi-tenor JPY curve out of one point, this module
sources that one real point and lets the caller apply it flat across
tenors (a documented approximation -- the actual JPY term structure has
shape too, just not freely available), OR fall back to
build_market.py's zero-differential path when even that is undesired.
"""
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
JAPAN_10Y_SERIES = "IRLTLT01JPM156N"  # Japan long-term (10Y) government bond yield, monthly


def fetch_jpy_10y_yield() -> float:
    """Latest Japan 10Y government bond yield, as a decimal (FRED publishes percent)."""
    resp = requests.get(FRED_CSV_URL.format(series_id=JAPAN_10Y_SERIES), timeout=15)
    resp.raise_for_status()
    lines = [l for l in resp.text.strip().splitlines() if l and "," in l]
    for line in reversed(lines[1:]):
        _, value = line.split(",", 1)
        value = value.strip()
        if value not in ("", "."):
            return float(value) / 100.0
    raise ValueError(f"no valid observation found for FRED series {JAPAN_10Y_SERIES}")


def flat_jpy_rate_by_tenor(tenors_years, rate: float = None) -> dict:
    """{tenor: rate} using ONE sourced rate (default: the 10Y JGB yield)
    applied flat across every requested tenor -- documented approximation,
    see module docstring."""
    if rate is None:
        rate = fetch_jpy_10y_yield()
    return {t: rate for t in tenors_years}
