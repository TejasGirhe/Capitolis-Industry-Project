"""
USD OIS (SOFR) curve pillars -- sourced from FRED (Federal Reserve Economic
Data), free, no API key required.

Honest limitation: FRED does not publish bank-quoted SOFR OIS swap rates
(those are an interbank product, not free public data). What it DOES publish
-- overnight SOFR plus on-the-run Treasury/T-bill yields -- is the standard
free substitute for a USD curve term structure, and is what this module
sources. If you have access to a real OIS swap curve (Bloomberg, a prime
broker feed, etc.), prefer that; this is documented here as a proxy, not
represented as the literal SOFR OIS curve MARKET_DATA.md Sec.2.1 describes.

Where this plugs in: capitolis_pricers.curves.zero_curve(ref_date, tenors,
zero_rates) -- the pillars returned here are (tenor_years, rate) pairs ready
to hand straight to that builder, e.g.:

    from risk_engine.market_data import fetch_usd_curve_pillars
    from capitolis_pricers.curves import zero_curve

    tenors, rates = fetch_usd_curve_pillars()
    usd_curve = zero_curve(ref_date, tenors, rates)
"""
from ._fred import fetch_latest_fred_value

# (tenor in years, FRED series id) -- SOFR overnight anchors the short end;
# Treasury/T-bill yields (par, not zero, but close enough at these tenors for
# a curve-building proxy) extend it out to 10Y.
USD_CURVE_SERIES = [
    (1 / 365, "SOFR"),      # overnight SOFR
    (0.25, "DTB3"),         # 3M T-bill (discount basis)
    (0.5, "DTB6"),          # 6M T-bill (discount basis)
    (1.0, "DGS1"),          # 1Y Treasury (CMT, par yield)
    (2.0, "DGS2"),
    (5.0, "DGS5"),
    (10.0, "DGS10"),
]


def fetch_usd_curve_pillars():
    """Returns (tenors_years, zero_rates) -- ready for
    capitolis_pricers.curves.zero_curve(ref_date, tenors, zero_rates).

    T-bill rates (DTB3/DTB6) are quoted on a discount basis by FRED, not as
    a bond-equivalent/zero yield; the difference is small at 3-6M tenors and
    is treated here as a proxy, not corrected for discount-vs-yield
    convention -- documented, not silently assumed precise.
    """
    tenors, rates = [], []
    for tenor, series_id in USD_CURVE_SERIES:
        rate = fetch_latest_fred_value(series_id)
        tenors.append(tenor)
        rates.append(rate)
    return tenors, rates
