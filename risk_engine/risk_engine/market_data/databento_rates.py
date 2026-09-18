"""
USD short-rate volatility SURFACE (term structure AND smile) sourced from
real, listed SOFR futures options on CME Globex (Databento's GLBX.MDP3
dataset), replacing the flat rate-vol placeholder every other part of this
project has used to date (see _sourced_book.py's BASE_RATE_VOL) -- no free
source for the true OTC swaption/cap-floor vol surface exists (confirmed:
Databento's own dataset catalog is entirely exchange/venue-fed, and
swaptions/caps are OTC-traded, not exchange-listed -- see this module's
project-memory note for the direct investigation), so SOFR futures options
are the standard, liquid, listed PROXY the market itself uses for the front
end of the USD rate-vol surface when true swaption vol isn't accessible.

Product mechanics (confirmed directly against Databento's own definition/
statistics schemas, not assumed):
    - Symbol root SR3 = 3-month SOFR futures options on CME Globex.
    - Quoted on a price scale of 100 - rate (e.g. price 95.91 => an implied
      3-month SOFR level of 4.09%), same convention as Eurodollar options.
    - `SR3.OPT` (Databento "parent" symbology) resolves every SOFR-option
      instrument; `definition` schema gives strike_price/expiration/
      instrument_class/underlying per instrument; `statistics` schema with
      stat_type == SETTLEMENT_PRICE (3) gives the OFFICIAL daily settlement
      price for EVERY strike (dense -- unlike raw trades, which are thin
      for far-OTM strikes on any single day).

VOL MODEL: Bachelier (normal), not Black-76 (lognormal) -- the market-
standard convention for SOFR/rates options, since the underlying rate can
be low or (historically, for some rates) negative, which breaks a
lognormal model's support. See bachelier_implied_vol below for the
closed-form-inversion formula used.

FULL SMILE, not ATM-only: risk_engine.calibration.market_surface.VolSurface
already supports a tenor x strike grid (bilinear interpolation), and
risk_engine.models._shared.fit_skew_smile already consumes >=3 strikes at
the longest tenor to fit real LGM-SV skew (rho) and curvature (eta)
parameters -- confirmed by reading that function before choosing to extract
the full smile here rather than only an ATM term structure, so this is not
wasted extraction: it directly improves LGM1F_SV/LGM2F_SV calibration,
which today falls back to a literature-prior rho/eta with no smile data at
all. fit_skew_smile expects MONEYNESS-style strikes centered near 1.0 (its
own code: `k_arr = np.array(ks) - 1.0`), so strikes here are normalized to
(rate-scale strike) / (underlying future's own rate-scale price) before
being handed to VolSurface, NOT left as raw absolute strikes like 9825.0.

    python -m risk_engine.market_data.databento_rates   (smoke test)
"""
import math
import os
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from ..calibration.market_surface import VolSurface

SOFR_OPTION_ROOT = "SR3.OPT"
DATASET = "GLBX.MDP3"
# Databento's `definition` schema reports strike_price as a raw integer-like
# float 100x the true rate-price scale (e.g. strike_price=9743.0 means a
# TRUE strike of 97.43 on the SAME 100-minus-rate scale the underlying
# future itself quotes at, e.g. a future close of 95.91). SETTLEMENT_PRICE
# statistics are on that SAME raw x100 scale as strike_price, NOT the
# future's own already-human-scale ohlcv close. Confirmed directly, not
# assumed: for SR3U5 P9743 (a put struck at true level 97.43, future at
# 95.91 => intrinsic = 97.43-95.91 = 1.52), the observed settlement price
# was 154.00 -- dividing by 100 gives 1.54, i.e. 0.02 of time value on a
# ~65-business-day option, which is a sane, small time-value figure; NOT
# dividing (treating 154.00 as already the true price) would imply 152.48
# points of time value on a sub-2-point intrinsic option, which is not a
# plausible options price under any model. This constant converts every
# raw (strike_price, settlement price) pair to the TRUE rate-price scale
# that matches the future's own ohlcv close directly.
RAW_TO_TRUE_SCALE = 1.0 / 100.0
CONTRACT_MULTIPLIER_YEARS = 0.25   # SR3 = 3-month SOFR -- rate sensitivity per contract is quarterly


def _client():
    import databento as db
    return db.Historical()


def bachelier_price(is_call: bool, F: float, K: float, T: float, sigma_n: float) -> float:
    """Bachelier (normal-model) forward option price, undiscounted.
    F, K in the SAME units as sigma_n (here: 100x(100-rate) price points).
    sigma_n: normal vol in price-point terms (annualized)."""
    if T <= 0 or sigma_n <= 0:
        return max(F - K, 0.0) if is_call else max(K - F, 0.0)
    d = (F - K) / (sigma_n * math.sqrt(T))
    if is_call:
        return (F - K) * norm.cdf(d) + sigma_n * math.sqrt(T) * norm.pdf(d)
    return (K - F) * norm.cdf(-d) + sigma_n * math.sqrt(T) * norm.pdf(d)


def bachelier_implied_vol(is_call: bool, price: float, F: float, K: float, T: float) -> Optional[float]:
    """Inverts bachelier_price for sigma_n via Brent's method. Returns None
    if the price is outside the model's achievable range (e.g. a stale or
    crossed settlement print) rather than returning a nonsensical vol --
    callers should skip that (tenor, strike) point, not propagate a bad
    number into the surface."""
    if T <= 0:
        return None
    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
    if price < intrinsic - 1e-9:
        return None   # below intrinsic value -- not a valid option price (stale/crossed print)

    def _obj(sigma):
        return bachelier_price(is_call, F, K, T, sigma) - price

    lo, hi = 1e-6, 500.0   # price-point vol search range -- wide enough for any realistic SOFR option
    try:
        if _obj(lo) > 0:
            return None   # price below even near-zero-vol intrinsic+time-value -- shouldn't happen post the check above
        if _obj(hi) < 0:
            return None   # price implausibly large even at very high vol -- bad print, skip
        return brentq(_obj, lo, hi)
    except ValueError:
        return None


def _year_frac(d0: date, d1: date) -> float:
    return max((d1 - d0).days / 365.0, 0.0)


def fetch_sofr_option_smile(as_of: date, max_tenor_years: float = 2.0,
                             min_strikes_per_tenor: int = 5) -> VolSurface:
    """Builds a real tenor x moneyness VolSurface for the 'RATE_USD' factor
    from CME SOFR futures options settlement prices, as of `as_of`.

    Steps, all against REAL Databento data (no synthetic fallback -- if the
    feed has nothing usable for a tenor, that tenor is simply absent from
    the returned surface rather than filled with a guess):
      1. Pull SR3.OPT `definition` records active as of `as_of` to get each
         instrument's strike_price / expiration / instrument_class / underlying.
      2. Pull SR3.OPT `statistics` (stat_type == SETTLEMENT_PRICE) for the
         most recent settlement AT OR BEFORE `as_of`.
      3. Pull the underlying futures' own settlement prices (schema
         ohlcv-1d close, or statistics settlement) to get F per expiry group.
      4. For each (underlying, strike) pair with a valid settlement price,
         Bachelier-invert to a normal implied vol; OTM option convention
         (calls above F, puts below) is used to avoid low-liquidity ITM prints.
      5. Normalize strike to moneyness (strike / F) and tenor to
         year-fraction from `as_of`, keep tenors with >= min_strikes_per_tenor
         usable points (fewer than that isn't a real smile -- see
         fit_skew_smile's own >=3-strikes floor; 5 is used here for a
         margin of safety against a couple of points failing inversion).

    Raises RuntimeError if NO usable tenor survives -- callers must decide
    explicitly how to fall back (e.g. flat_vol_surface), this function
    itself never silently substitutes a placeholder.
    """
    client = _client()
    start = as_of - timedelta(days=5)   # small lookback window in case as_of itself has no settlement yet
    end = as_of + timedelta(days=1)

    defs = client.timeseries.get_range(
        dataset=DATASET, symbols=[SOFR_OPTION_ROOT], stype_in="parent",
        schema="definition", start=start.isoformat(), end=end.isoformat(),
    ).to_df()
    if defs.empty:
        raise RuntimeError(f"no SR3.OPT definitions returned for window ending {as_of}")
    defs = defs[defs["instrument_class"].isin(["C", "P"])].copy()
    defs["expiration_date"] = defs["expiration"].dt.date

    stats = client.timeseries.get_range(
        dataset=DATASET, symbols=[SOFR_OPTION_ROOT], stype_in="parent",
        schema="statistics", start=start.isoformat(), end=end.isoformat(),
    ).to_df()
    settle = stats[stats["stat_type"] == 3].copy()   # SETTLEMENT_PRICE
    if settle.empty:
        raise RuntimeError(f"no SR3.OPT settlement-price statistics returned for window ending {as_of}")
    # keep the LATEST settlement print per symbol (definition/statistics
    # publish preliminary-then-final settlement events on the same day --
    # confirmed directly, duplicate timestamps per symbol in the raw feed)
    settle = settle.sort_values("ts_event").groupby("symbol", as_index=False).last()
    settle_by_symbol = dict(zip(settle["symbol"], settle["price"]))

    underlyings = defs["underlying"].unique().tolist()
    fut = client.timeseries.get_range(
        dataset=DATASET, symbols=underlyings, schema="ohlcv-1d",
        start=start.isoformat(), end=end.isoformat(),
    ).to_df()
    if fut.empty:
        raise RuntimeError(f"no underlying SR3 futures OHLCV returned for window ending {as_of}")
    fut = fut.sort_values("ts_event").groupby("symbol", as_index=False).last()
    fut_price_by_symbol = dict(zip(fut["symbol"], fut["close"]))

    points = defaultdict(list)   # tenor_years -> [(moneyness, vol), ...]
    for _, row in defs.iterrows():
        sym = row["raw_symbol"]
        if sym not in settle_by_symbol:
            continue
        underlying = row["underlying"]
        if underlying not in fut_price_by_symbol:
            continue
        F = float(fut_price_by_symbol[underlying])   # ohlcv close is already on the true rate-price scale
        K = float(row["strike_price"]) * RAW_TO_TRUE_SCALE
        expiry = row["expiration_date"]
        T = _year_frac(as_of, expiry)
        if T <= 0 or T > max_tenor_years:
            continue
        is_call = row["instrument_class"] == "C"
        # OTM-only convention (standard smile-construction practice): use
        # calls struck above the forward, puts struck below -- these are
        # the more liquid, more reliably-quoted side of the market.
        if is_call and K < F:
            continue
        if not is_call and K > F:
            continue
        price = float(settle_by_symbol[sym]) * RAW_TO_TRUE_SCALE   # settlement price is on the SAME raw scale as strike_price
        vol = bachelier_implied_vol(is_call, price, F, K, T)
        if vol is None:
            continue
        moneyness = K / F
        points[round(T, 4)].append((moneyness, vol))

    surface_tenors, surface_strikes = set(), set()
    vols: Dict = {}
    for T, pts in points.items():
        if len(pts) < min_strikes_per_tenor:
            continue
        for moneyness, vol in pts:
            k = round(moneyness, 4)
            surface_tenors.add(T)
            surface_strikes.add(k)
            vols[(T, k)] = vol   # last write wins on a rare exact-duplicate moneyness bucket

    if not surface_tenors:
        raise RuntimeError(
            f"no tenor had >= {min_strikes_per_tenor} usable Bachelier-inverted strikes as of {as_of} "
            "-- SOFR options data was present but too sparse to build a real smile for this date")

    # VolSurface.vol() bilinearly interpolates and needs the SAME strike
    # grid defined at every tenor -- real market data won't naturally align
    # like that (different tenors quote different strike increments), so
    # missing (tenor, strike) cells are filled by nearest-available-tenor
    # moneyness interpolation within that tenor's own points, keeping this
    # a real-data derived fill, not an invented number.
    surface_tenors = sorted(surface_tenors)
    surface_strikes = sorted(surface_strikes)
    for T in surface_tenors:
        pts = sorted(points[round(T, 4)]) if round(T, 4) in points else []
        if not pts:
            continue
        ks_have = np.array([p[0] for p in pts])
        vs_have = np.array([p[1] for p in pts])
        for k in surface_strikes:
            if (T, k) in vols:
                continue
            vols[(T, k)] = float(np.interp(k, ks_have, vs_have))

    return VolSurface(factor_key="RATE_USD", tenors=surface_tenors, strikes=surface_strikes, vols=vols)


if __name__ == "__main__":
    import sys
    as_of = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today() - timedelta(days=3)
    surf = fetch_sofr_option_smile(as_of)
    print(f"Built RATE_USD vol surface as of {as_of}: {len(surf.tenors)} tenors, {len(surf.strikes)} strikes")
    print("Tenors (years):", [round(t, 3) for t in surf.tenors])
    print("ATM term structure:", [(round(t, 3), round(v, 4)) for t, v in surf.atm_term_structure()])
