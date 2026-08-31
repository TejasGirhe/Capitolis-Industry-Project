"""
Credit / risky bonds -- OPTIONAL (extension).

Bonds ship **risk-free** (government / agency). If the project is extended to
**risky bonds**, supply the bond ISSUER's credit curve and the bond is priced on
a credit-risky basis: cash flows are survival-weighted and a recovery is paid on
default. The bond's credit charge (its "CVA") is then

    credit charge = risk-free price - credit-risky price

This is issuer credit on the underlying bond -- not counterparty credit. It is
off unless an issuer `CreditCurve` is supplied in `MarketState.credit_curves`
(keyed by the bond's `issuer`); with none, bonds price risk-free as before.
"""

import math
from .daycount import to_date, year_fraction


class CreditCurve:
    """Issuer survival curve from CDS spreads via the credit triangle.

    Approximation: hazard(t) = spread(t) / (1 - recovery); survival
    S(t) = exp(-hazard(t) * t), spread linearly interpolated in t. Replace with a
    proper CDS bootstrap for production.
    """

    def __init__(self, ref_date, tenors_years, cds_spreads, recovery=0.40):
        self.ref_date = to_date(ref_date)
        self.recovery = float(recovery)
        pts = sorted(zip([float(t) for t in tenors_years],
                         [float(s) for s in cds_spreads]))
        self._t = [p[0] for p in pts]
        self._s = [p[1] for p in pts]

    def _spread(self, t):
        ts, ss = self._t, self._s
        if t <= ts[0]:
            return ss[0]
        if t >= ts[-1]:
            return ss[-1]
        for i in range(1, len(ts)):
            if t <= ts[i]:
                w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
                return ss[i - 1] * (1 - w) + ss[i] * w

    def survival(self, d):
        """Survival probability to date d."""
        t = year_fraction(self.ref_date, to_date(d), "ACT/365F")
        if t <= 0:
            return 1.0
        hazard = self._spread(t) / (1.0 - self.recovery)
        return math.exp(-hazard * t)

    def default_prob(self, d1, d2):
        """Marginal default probability in (d1, d2]."""
        return self.survival(d1) - self.survival(d2)
