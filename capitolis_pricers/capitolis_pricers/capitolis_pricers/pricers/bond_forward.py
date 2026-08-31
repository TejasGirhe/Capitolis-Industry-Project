"""
Bond Forward.

NPV = sign * units * (par/100) * DF(T) * (forward_clean(T) - strike_clean)

The forward clean price is either:
  * repo-financed (when `repo_rate` is given): the current dirty price carried to
    the forward date at the repo rate (ACT/360), less any coupons paid in the
    window (each also carried at repo), minus accrued at T. This is the desk
    convention (spot financed at repo). Risk-free (govie/agency) bonds only.
  * curve-derived (when `repo_rate` is blank): forward implied by the discount
    curve. Used automatically in simulation, where a future price can't be
    observed.
"""
from .base import Pricer
from ..market import MarketState
from ..bond import FixedRateBond
from ..daycount import to_date, year_fraction


class BondForwardTrade(Pricer):
    def __init__(self, bond, forward_date, strike_clean, position="long",
                 trade_ccy="USD", notional=0.0, repo_rate=None):
        self.bond = bond
        self.forward_date = to_date(forward_date)
        self.strike_clean = float(strike_clean)
        self.position = (position or "").strip().lower()
        if self.position not in ("long", "short"):
            raise ValueError(f"position must be 'long' or 'short', got {position!r}")
        self.trade_currency = trade_ccy.upper()
        self.notional = float(notional)
        if self.notional <= 0:
            raise ValueError("notional must be positive; use position=long/short for the side")
        self.repo_rate = None if repo_rate in (None, "") else float(repo_rate)

    @classmethod
    def from_terms(cls, bond_terms, **kw):
        return cls(FixedRateBond(**bond_terms), **kw)

    def _credit(self, market):
        return market.credit(self.bond.issuer) if self.bond.issuer else None

    def _forward_clean(self, market):
        disc = market.discount(self.trade_currency)
        credit = self._credit(market)               # None unless issuer curve supplied
        if self.repo_rate is None:
            return self.bond.forward_clean(disc, self.forward_date, credit)  # curve-derived
        # repo-financed: carry the current dirty price to T at the repo rate
        ref, T = market.ref_date, self.forward_date
        spot_dirty = self.bond.dirty_price(disc, ref, credit)
        fwd_dirty = spot_dirty * (1.0 + self.repo_rate * year_fraction(ref, T, "ACT/360"))
        for pay, amt in self.bond.cashflows():
            if ref < pay <= T and pay != self.bond.maturity:            # coupons in the window
                fwd_dirty -= amt * (1.0 + self.repo_rate * year_fraction(pay, T, "ACT/360"))
        return fwd_dirty - self.bond.accrued(T)

    def _npv_trade_ccy(self, market: MarketState) -> float:
        disc = market.discount(self.trade_currency)
        fwd_clean = self._forward_clean(market)
        df = disc.discount(self.forward_date)
        npv_long = (self.notional / self.bond.face) * df * (fwd_clean - self.strike_clean)
        return npv_long if self.position == "long" else -npv_long

    def forward_clean_price(self, market: MarketState) -> float:
        return self._forward_clean(market)

    def describe(self):
        carry = (f"repo {self.repo_rate*100:.3f}%" if self.repo_rate is not None
                 else "curve carry")
        return (f"Bond Forward ({self.position}) on {self.bond.coupon*100:.3f}% "
                f"bond maturing {self.bond.maturity}, notional {self.notional:,.0f}, "
                f"settle {self.forward_date}, strike {self.strike_clean:.4f} "
                f"{self.trade_currency}, {carry}")
