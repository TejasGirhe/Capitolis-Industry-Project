"""
Extended volatility surface: tenor x strike/moneyness, one per risk factor.

MARKET_DATA.md Sec.5 (Capitolis-provided) specifies one volatility per
(factor, tenor) -- a term structure only, no strike axis. This engine's
stochastic-vol models (LGM1F_SV, LGM2F_SV) need a smile to have identifiable
vol-of-vol / correlation parameters, so the collection spec is extended here
with a `strike` (rates: absolute strike) or `moneyness` (equity/FX: K/F)
column. This is risk_engine's own spec, layered on top of -- not editing --
the Capitolis-provided MARKET_DATA.md.

CSV columns: factor, tenor, strike_or_moneyness, volatility
    factor: 'RATE_USD', an equity ISIN, or 'FX_USDJPY' (matches
        str(RateFactor(...))/str(EquityFactor(...))/str(FxFactor(...)))
    tenor: expiry in years (float) or a tenor label ('1Y', '2Y', ...)
    strike_or_moneyness: absolute strike (rates) or moneyness K/F (equity/FX)
    volatility: decimal (0.20 = 20%)
"""
import csv
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_TENOR_YEARS = {
    "1M": 1 / 12, "2M": 2 / 12, "3M": 3 / 12, "6M": 0.5, "9M": 0.75,
    "1Y": 1.0, "18M": 1.5, "2Y": 2.0, "3Y": 3.0, "4Y": 4.0, "5Y": 5.0,
    "7Y": 7.0, "10Y": 10.0,
}


def tenor_to_years(t):
    """Accepts a known label from _TENOR_YEARS (kept for backward
    compatibility with existing callers), a general '<n><W|M|Y>' label
    (e.g. '1W', '4M', '21M', '35Y' -- real Bloomberg SWPM/VCUB tenor labels
    this project did not previously need to parse, confirmed against the
    actual data_bloomberg files rather than a guessed superset), or a bare
    number (already in years)."""
    if isinstance(t, str):
        s = t.strip().upper()
        if s in _TENOR_YEARS:
            return _TENOR_YEARS[s]
        if len(s) >= 2 and s[-1] in ("D", "W", "M", "Y") and s[:-1].replace(".", "", 1).isdigit():
            n = float(s[:-1])
            unit_years = {"D": 1 / 365.0, "W": 7 / 365.0, "M": 1 / 12, "Y": 1.0}[s[-1]]
            return n * unit_years
    return float(t)


@dataclass
class VolSurface:
    """A single factor's tenor x strike grid, bilinearly interpolated,
    flat-extrapolated at the edges."""
    factor_key: str
    tenors: List[float] = field(default_factory=list)         # sorted, years
    strikes: List[float] = field(default_factory=list)        # sorted, absolute or moneyness
    vols: Dict[Tuple[float, float], float] = field(default_factory=dict)  # (tenor, strike) -> vol

    def atm_term_structure(self) -> List[Tuple[float, float]]:
        """(tenor, vol) pairs at the strike closest to ATM (moneyness/strike == 1.0
        for equity/FX moneyness grids, or the median strike for a rate grid)."""
        if not self.strikes:
            return []
        atm_strike = min(self.strikes, key=lambda k: abs(k - 1.0)) if any(
            abs(k - 1.0) < 5.0 for k in self.strikes) else self.strikes[len(self.strikes) // 2]
        return [(t, self.vol(t, atm_strike)) for t in self.tenors]

    def _bracket(self, x, xs):
        if x <= xs[0]:
            return xs[0], xs[0], 0.0
        if x >= xs[-1]:
            return xs[-1], xs[-1], 0.0
        i = bisect_left(xs, x)
        lo, hi = xs[i - 1], xs[i]
        w = (x - lo) / (hi - lo) if hi > lo else 0.0
        return lo, hi, w

    def vol(self, tenor: float, strike: float) -> float:
        """Bilinear interpolation in (tenor, strike); flat extrapolation at edges."""
        if not self.tenors or not self.strikes:
            raise ValueError(f"empty vol surface for {self.factor_key}")
        t_lo, t_hi, wt = self._bracket(tenor, self.tenors)
        k_lo, k_hi, wk = self._bracket(strike, self.strikes)
        v_ll = self.vols[(t_lo, k_lo)]
        v_lh = self.vols[(t_lo, k_hi)]
        v_hl = self.vols[(t_hi, k_lo)]
        v_hh = self.vols[(t_hi, k_hi)]
        v_lo = v_ll * (1 - wk) + v_lh * wk
        v_hi = v_hl * (1 - wk) + v_hh * wk
        return v_lo * (1 - wt) + v_hi * wt


def load_vol_surface(path: str, factor_key: str) -> VolSurface:
    """Load one factor's slice of a tenor x strike vol CSV (see module docstring
    for columns). Rows for other factor values are skipped."""
    tenors, strikes = set(), set()
    vols = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["factor"].strip() != factor_key:
                continue
            t = tenor_to_years(row["tenor"].strip())
            k = float(row["strike_or_moneyness"])
            v = float(row["volatility"])
            tenors.add(t)
            strikes.add(k)
            vols[(t, k)] = v
    if not vols:
        raise KeyError(f"no vol rows found for factor '{factor_key}' in {path}")
    return VolSurface(factor_key=factor_key, tenors=sorted(tenors), strikes=sorted(strikes), vols=vols)


def flat_vol_surface(factor_key: str, flat_vol: float, tenors=(0.5, 1.0, 2.0, 5.0, 10.0)) -> VolSurface:
    """A degenerate flat surface (single strike, no smile) -- useful for sanity
    tests and for factors where only a term structure (no smile) is available."""
    strikes = [1.0]
    vols = {(t, 1.0): flat_vol for t in tenors}
    return VolSurface(factor_key=factor_key, tenors=sorted(tenors), strikes=strikes, vols=vols)
