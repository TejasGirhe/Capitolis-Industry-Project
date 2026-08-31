"""
Fixed-rate bond -- cashflows, dirty/clean price, accrued, and forward price.

Prices are quoted per 100 face (market convention). All discounted-cash-flow
maths; no third-party dependency. Reproduces a reference implementation to the
sixth decimal.
"""

from .daycount import to_date, year_fraction, schedule_backward


class FixedRateBond:
    def __init__(self, issue_date, maturity_date, coupon, frequency_months,
                 day_count="30/360", face=100.0, issuer=None):
        self.issuer = issuer
        self.issue = to_date(issue_date)
        self.maturity = to_date(maturity_date)
        self.coupon = float(coupon)
        self.freq_m = int(frequency_months)
        self.dcc = day_count
        self.face = float(face)
        self._boundaries = schedule_backward(self.issue, self.maturity, self.freq_m)

    def cashflows(self):
        """List of (pay_date, amount) per `face`, coupons + redemption."""
        cfs = []
        b = self._boundaries
        for start, end in zip(b[:-1], b[1:]):
            amt = self.face * self.coupon * year_fraction(start, end, self.dcc)
            cfs.append((end, amt))
        # redemption at maturity (added to the final coupon's pay date)
        cfs.append((self.maturity, self.face))
        return cfs

    def _accrual_period(self, d):
        d = to_date(d)
        b = self._boundaries
        for start, end in zip(b[:-1], b[1:]):
            if start <= d < end:
                return start, end
        return None

    def accrued(self, settle_date):
        p = self._accrual_period(settle_date)
        if p is None:
            return 0.0
        start, end = p
        return self.face * self.coupon * year_fraction(start, settle_date, self.dcc)

    def dirty_price(self, curve, settle_date, credit=None):
        """PV of cashflows after settle, expressed at settle (per face).
        If `credit` (issuer CreditCurve) is given, price credit-risky:
        survival-weight cashflows and add recovery-on-face at default."""
        settle = to_date(settle_date)
        df_s = curve.discount(settle)
        cfs = [(p, a) for p, a in self.cashflows() if p > settle]
        if credit is None:
            return sum(a * curve.discount(p) for p, a in cfs) / df_s
        pv = sum(a * curve.discount(p) * credit.survival(p) for p, a in cfs)
        prev = settle                                   # recovery on face at default
        for p in sorted({d for d, _ in cfs}):
            pv += credit.recovery * self.face * curve.discount(p) * \
                  (credit.survival(prev) - credit.survival(p))
            prev = p
        return pv / df_s

    def clean_price(self, curve, settle_date):
        return self.dirty_price(curve, settle_date) - self.accrued(settle_date)

    def forward_dirty(self, curve, forward_date, credit=None):
        """Forward dirty price to `forward_date` (per face)."""
        T = to_date(forward_date)
        if credit is None:
            return sum(a * curve.discount(p) for p, a in self.cashflows() if p > T) / curve.discount(T)
        return self.dirty_price(curve, T, credit) * 1.0   # dirty at T via risky PV

    def forward_clean(self, curve, forward_date, credit=None):
        return self.forward_dirty(curve, forward_date, credit) - self.accrued(forward_date)
