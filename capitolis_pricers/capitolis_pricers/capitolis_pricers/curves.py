"""
Interest-rate and FX curves -- standard library only.

A `Curve` is a discount-factor curve: it maps a date to a discount factor
DF(0, t). Interpolation is log-linear on the discount factors (equivalently
piecewise-flat instantaneous forwards), the market/ORE default that keeps
forwards positive. Curve time is year-fraction on ACT/365F from the reference
date.

These builders take plain numbers as inputs and have no third-party
dependency, so they can be handed to students as-is and fed into their own
Monte Carlo engine (rebuild a Curve per simulated node, or shift the nodes).
"""

import math
from .daycount import to_date, year_fraction


class Curve:
    """Discount curve defined by pillar times (years) and discount factors."""

    def __init__(self, ref_date, pillar_times, discount_factors, basis="ACT/365F"):
        self.ref_date = to_date(ref_date)
        self.basis = basis
        # store as sorted (t, ln DF); ensure t=0 -> DF=1 anchor
        pts = sorted(zip(pillar_times, discount_factors))
        if pts[0][0] > 0.0:
            pts = [(0.0, 1.0)] + pts
        self._t = [p[0] for p in pts]
        self._lndf = [math.log(p[1]) for p in pts]

    def _time(self, d):
        return year_fraction(self.ref_date, to_date(d), self.basis)

    def _ln_df_at_t(self, t):
        ts, ys = self._t, self._lndf
        if t <= ts[0]:
            # flat zero-rate extrapolation at the short end
            slope = ys[1] / ts[1] if ts[1] > 0 else 0.0
            return slope * t
        if t >= ts[-1]:
            # flat zero-rate extrapolation at the long end
            z = -ys[-1] / ts[-1]
            return -z * t
        # locate bracketing pillars and linearly interpolate ln DF
        lo, hi = 0, len(ts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        w = (t - ts[lo]) / (ts[hi] - ts[lo])
        return ys[lo] * (1 - w) + ys[hi] * w

    def discount(self, d):
        """Discount factor DF(ref, d)."""
        return math.exp(self._ln_df_at_t(self._time(d)))

    def zero_rate(self, d, comp="continuous"):
        t = self._time(d)
        if t <= 0:
            return 0.0
        return -self._ln_df_at_t(t) / t


# ------------------------------------------------------------ builders
def flat_curve(ref_date, rate, basis="ACT/365F"):
    """Flat continuously-compounded curve."""
    ref = to_date(ref_date)
    # two pillars are enough given log-linear interpolation
    return Curve(ref, [1.0, 30.0], [math.exp(-rate * 1.0), math.exp(-rate * 30.0)], basis)


def zero_curve(ref_date, tenors_years, zero_rates, basis="ACT/365F"):
    """Curve from continuously-compounded zero-rate nodes."""
    dfs = [math.exp(-z * t) for z, t in zip(zero_rates, tenors_years)]
    return Curve(ref_date, list(tenors_years), dfs, basis)


class FxCurve:
    """FX by covered-interest parity: F(0,T) = S * DF_base(T) / DF_quote(T).

    Quoted as QUOTE currency per 1 BASE currency, e.g. base='USD', quote='JPY'
    -> USDJPY (~150 JPY per USD).
    """

    def __init__(self, base_ccy, quote_ccy, spot, base_curve=None, quote_curve=None):
        self.base_ccy = base_ccy.upper()
        self.quote_ccy = quote_ccy.upper()
        self.spot = float(spot)
        self.base_curve = base_curve
        self.quote_curve = quote_curve

    def forward(self, d):
        if self.base_curve is None or self.quote_curve is None:
            raise ValueError("FX forwards need both discount curves; only spot is set. "
                             "Build the FX forward curve from FX swap points (see "
                             "MARKET_DATA.md).")
        return self.spot * self.base_curve.discount(d) / self.quote_curve.discount(d)

    def convert(self, amount, from_ccy, to_ccy):
        from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()
        if from_ccy == to_ccy:
            return amount
        if (from_ccy, to_ccy) == (self.base_ccy, self.quote_ccy):
            return amount * self.spot
        if (from_ccy, to_ccy) == (self.quote_ccy, self.base_ccy):
            return amount / self.spot
        raise ValueError(f"FxCurve covers {self.base_ccy}/{self.quote_ccy}")
