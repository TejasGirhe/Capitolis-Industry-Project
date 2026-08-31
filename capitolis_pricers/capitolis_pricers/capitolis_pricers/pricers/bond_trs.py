"""
Bond Total Return Swap.

The total-return receiver gets the bond's total return (price change +
coupons); the payer gets a funding leg (overnight index + spread). Decomposed
into a return leg and a funding leg over the reset schedule (as ORE does).

Return leg, per reset period [t_i, t_{i+1}] (per `units`):
    units * ( P_dirty(t_{i+1}) - P_dirty(t_i) )  +  coupons paid in (t_i, t_{i+1}]
where P_dirty(t) is the forward dirty price from the curve. Over a period that
brackets a coupon the dirty price drops by the coupon while the coupon is added
back, so the two net to the true capital + income return. Discounted at DF(t_{i+1}).

Funding leg, per period:
    (simple_fwd + spread) * funding_notional * tau * DF(t_{i+1})

NPV(receiver) = return leg - funding leg.

Modelling choices to state in the report: deterministic-curve forward prices are
used for E[P_dirty(t)] (the interim financing convexity is captured by your
stochastic engine when it reprices the bond on simulated paths -- this is the
analytic benchmark it converges to); funding_notional defaults to the initial
dirty market value, held constant (a resetting notional is a common variant);
the income/repo curve equals the discount curve.
"""

from .base import Pricer
from ..market import MarketState
from ..bond import FixedRateBond
from ..daycount import to_date, year_fraction, schedule_forward


class BondTRS(Pricer):
    def __init__(self, bond, notional, start_date, end_date,
                 reset_frequency_months=3, spread=0.0, fixed_rate=None,
                 funding_notional=None, direction="pay_tr", trade_ccy="USD"):
        self.bond = bond
        self.notional = float(notional)
        if self.notional <= 0:
            raise ValueError("notional must be positive; use direction for the side")
        self.units = float(notional) / float(bond.face)  # number of par-face bonds
        self.start_date = to_date(start_date)
        self.end_date = to_date(end_date)
        self.reset_m = int(reset_frequency_months)
        self.spread = float(spread)
        # funding is FIXED when a fixed_rate is supplied, else FLOATING (SOFR+spread)
        self.fixed_rate = None if fixed_rate in (None, "") else float(fixed_rate)
        self.fixed = self.fixed_rate is not None
        self.funding_notional = (None if funding_notional in (None, "")
                                 else float(funding_notional))
        self.direction = (direction or "").strip().lower()
        if self.direction not in ("receive_tr", "pay_tr"):
            raise ValueError(f"direction must be 'receive_tr' or 'pay_tr', got {direction!r}")
        self.trade_currency = trade_ccy.upper()

    @classmethod
    def from_terms(cls, bond_terms, **kw):
        return cls(FixedRateBond(**bond_terms), **kw)

    def _credit(self, market):
        return market.credit(self.bond.issuer) if self.bond.issuer else None

    def _return_leg(self, market, disc, dates):
        credit = self._credit(market)
        cfs = self.bond.cashflows()
        boundaries = self.bond._boundaries  # coupon accrual boundaries
        pv = 0.0
        for t0, t1 in zip(dates[:-1], dates[1:]):
            d_price = (self.bond.forward_dirty(disc, t1, credit)
                       - self.bond.forward_dirty(disc, t0, credit)) * self.units
            coupons = sum(a for (p, a) in cfs
                          if to_date(t0) < p <= to_date(t1) and p != self.bond.maturity
                          ) * self.units
            pv += disc.discount(t1) * (d_price + coupons)
        return pv

    def _funding_leg(self, disc, dates, notional):
        pv = 0.0
        for t0, t1 in zip(dates[:-1], dates[1:]):
            tau = year_fraction(t0, t1, "ACT/365F")
            if tau <= 0:
                continue
            df0, df1 = disc.discount(t0), disc.discount(t1)
            if self.fixed:
                rate = self.fixed_rate
            else:
                simple_fwd = (df0 / df1 - 1.0) / tau
                rate = simple_fwd + self.spread
            pv += rate * notional * tau * df1
        return pv

    def _npv_trade_ccy(self, market: MarketState) -> float:
        disc = market.discount(self.trade_currency)
        dates = schedule_forward(self.start_date, self.end_date, self.reset_m)
        fn = (self.funding_notional if self.funding_notional is not None
              else self.units * self.bond.dirty_price(disc, market.ref_date, self._credit(market)) / 100.0 * self.bond.face)
        receiver = self._return_leg(market, disc, dates) - self._funding_leg(disc, dates, fn)
        return receiver if self.direction == "receive_tr" else -receiver

    def describe(self):
        side = ("receive total return / pay funding" if self.direction == "receive_tr"
                else "pay total return / receive funding")
        return (f"Bond TRS on {self.bond.coupon*100:.3f}% bond maturing "
                f"{self.bond.maturity}: {side}, notional {self.notional:,.0f}, "
                f"spread {self.spread*1e4:.0f}bp")
