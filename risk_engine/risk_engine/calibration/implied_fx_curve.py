"""
Back out an implied JPY discount curve from FX forward points, for the FX
model's drift term only -- there is no JPY rate factor in this book
(MARKET_DATA.md Sec.2.2: "the JPY rate is embedded in those forwards but
never separately built"). This module builds that embedded rate explicitly,
as a deterministic curve, so FXGBM's risk-neutral drift
r_usd(t) - r_jpy_implied(t) has a well-defined foreign leg without inventing
a JPY rate factor to simulate.

Covered interest parity: F(0,T) = S * DF_USD(T) / DF_JPY(T)
    => DF_JPY(T) = S * DF_USD(T) / F(0,T)

capitolis_pricers.curves.FxCurve.forward(d) already implements the F(0,T)
side of this identity (base=USD, quote=JPY) -- but it requires BOTH curves to
already exist, which is circular for this use (we're trying to derive the
JPY curve). So this module works from raw forward POINTS (spot + tenor,
forward-price pillars), the actual MARKET_DATA.md Sec.2.2 collection format,
rather than from a pre-built FxCurve.
"""
from .market_surface import tenor_to_years
from capitolis_pricers.curves import Curve


def build_implied_jpy_curve(ref_date, fx_spot: float, forward_pillars, usd_curve) -> Curve:
    """
    fx_spot: USDJPY spot (JPY per USD).
    forward_pillars: list of (tenor, forward_price) pairs -- tenor as a label
        ('1M', '1Y', ...) or a float year fraction; forward_price is the
        outright USDJPY forward level at that tenor (spot + swap points,
        already converted to an outright rate, not raw points).
    usd_curve: capitolis_pricers.curves.Curve, the USD discount curve.

    Returns a Curve of DF_JPY(T) = fx_spot * DF_USD(T) / F(0,T) at each pillar,
    log-linear interpolated exactly like any other Curve here.
    """
    if not forward_pillars:
        raise ValueError("build_implied_jpy_curve needs at least one forward pillar "
                          "(spot + swap points, per MARKET_DATA.md Sec.2.2)")
    times, dfs = [], []
    for tenor, fwd in forward_pillars:
        t = tenor_to_years(tenor)
        df_usd = usd_curve.discount(_shift(ref_date, t))
        df_jpy = fx_spot * df_usd / fwd
        times.append(t)
        dfs.append(df_jpy)
    return Curve(ref_date, times, dfs)


def flat_implied_jpy_curve(ref_date, usd_curve) -> Curve:
    """Fallback when only FX spot (no forward points) is available: assume a
    zero USD/JPY rate differential, i.e. DF_JPY == DF_USD. Degenerates the FX
    drift to zero -- documented assumption, not a real calibration; prefer
    build_implied_jpy_curve whenever swap points are collected."""
    return usd_curve


def _shift(ref_date, years):
    from datetime import timedelta
    return ref_date + timedelta(days=round(years * 365.0))
