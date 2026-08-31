"""
Shared FRED (Federal Reserve Economic Data) fetch helper -- free, no API key
required. Used by rates.py (USD curve) and credit.py (rating-tier spread
proxy); factored out here so both modules share one fetch implementation
rather than duplicating the CSV-parsing logic.
"""
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_latest_fred_value(series_id: str) -> float:
    """Latest non-missing observation for a FRED series, as a decimal rate
    (FRED publishes percent, e.g. 4.30 -> 0.0430)."""
    resp = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=15)
    resp.raise_for_status()
    lines = [l for l in resp.text.strip().splitlines() if l and "," in l]
    for line in reversed(lines[1:]):  # skip header, walk backward for latest non-missing value
        _, value = line.split(",", 1)
        value = value.strip()
        if value not in ("", "."):
            return float(value) / 100.0
    raise ValueError(f"no valid observation found for FRED series {series_id}")
