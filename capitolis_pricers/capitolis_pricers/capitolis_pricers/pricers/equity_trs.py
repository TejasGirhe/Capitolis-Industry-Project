"""
Equity Total Return Swap -- single name or basket; USD and JPY (compo).

Assumes 100% dividend pass-through (a full TOTAL return swap). The equity
receiver gets the total return of the position; the payer gets a funding leg
(overnight index + spread). Shipped examples: CLGM pays funding (short it).

Position is SHARES + BASIS (strike/cost level per share) per constituent; the
notional weight is derived (shares*basis).

Equity leg (single total-return cash flow at maturity):

    equity leg NPV = sum_i shares_i * DF(end) * (F_i(end) - basis_i)
    F(t) = S_ccy / DF(end)   (= S_ccy * e^{r t}; grows at the funding rate)

At 100% pass-through the dividend rate NETS OUT (the ex-dividend price drag is
exactly offset by the dividends handed to the receiver), so the total-return
forward grows at r and the dividend rate does not enter the price. S_ccy is the
spot in the trade currency (compo: S_native * FX_spot).

Funding leg (periodic):
    sum (simple_fwd + spread) * N * tau * DF(t_{i+1}),
    N = funding_notional or sum shares_i*basis_i (initial position value).

NPV (equity receiver) = equity leg - funding leg.
"""

from .base import Pricer
from ..market import MarketState
from ..daycount import to_date, year_fraction, schedule_forward


class _Pos:
    __slots__ = ("isin", "shares", "basis", "currency")

    def __init__(self, isin, shares, basis, currency):
        self.isin = isin
        self.shares = float(shares)
        self.basis = None if basis in (None, "") else float(basis)
        self.currency = currency.upper()


class EquityTRS(Pricer):
    def __init__(self, positions, trade_ccy, start_date, end_date,
                 reset_frequency_months=3, spread=0.0,
                 direction="pay_equity", fixed_rate=None,
                 funding_notional=None, funding_index=None):
        self.positions = [
            _Pos(p["isin"], p["shares"], p.get("basis"), p["currency"])
            for p in positions]
        self.trade_currency = trade_ccy.upper()
        self.start_date = to_date(start_date)
        self.end_date = to_date(end_date)
        self.reset_m = int(reset_frequency_months)
        self.spread = float(spread)
        self.direction = (direction or "").strip().lower()
        if self.direction not in ("receive_equity", "pay_equity"):
            raise ValueError(f"direction must be 'receive_equity' or 'pay_equity', got {direction!r}")
        # funding is FIXED when a fixed_rate is supplied, else FLOATING (SOFR+spread)
        self.fixed_rate = None if fixed_rate in (None, "") else float(fixed_rate)
        self.fixed = self.fixed_rate is not None
        self.funding_notional = (None if funding_notional in (None, "")
                                 else float(funding_notional))
        if self.funding_notional is not None and self.funding_notional < 0:
            raise ValueError("funding_notional must be >= 0; use direction for the side")
        self.funding_index = funding_index

    def _spot_ccy(self, market, pos):
        return market.equity_spot(pos.isin) * market.fx(pos.currency, self.trade_currency)

    def _equity_leg(self, market):
        disc = market.discount(self.trade_currency)
        dfe = disc.discount(self.end_date)
        pv = 0.0
        for pos in self.positions:
            s_ccy = self._spot_ccy(market, pos)
            fwd = s_ccy / dfe                    # total-return forward (grows at r)
            basis = pos.basis if pos.basis is not None else s_ccy   # reset today if blank
            pv += pos.shares * dfe * (fwd - basis)
        return pv

    def _funding_notional_value(self, market):
        if self.funding_notional is not None:
            return self.funding_notional
        n = 0.0
        for pos in self.positions:
            basis = pos.basis if pos.basis is not None else self._spot_ccy(market, pos)
            n += pos.shares * basis
        return n

    def _funding_leg(self, market):
        disc = market.discount(self.trade_currency)
        notional = self._funding_notional_value(market)
        dates = schedule_forward(self.start_date, self.end_date, self.reset_m)
        pv = 0.0
        for t0, t1 in zip(dates[:-1], dates[1:]):
            if t1 <= market.ref_date:
                continue
            acc0 = max(t0, market.ref_date)
            tau = year_fraction(t0, t1, "ACT/365F")
            if tau <= 0:
                continue
            df0, df1 = disc.discount(acc0), disc.discount(t1)
            if self.fixed:
                rate = self.fixed_rate
            else:
                simple_fwd = (df0 / df1 - 1.0) / year_fraction(acc0, t1, "ACT/365F")
                rate = simple_fwd + self.spread
            pv += rate * notional * tau * df1
        return pv

    def _npv_trade_ccy(self, market: MarketState) -> float:
        receiver = self._equity_leg(market) - self._funding_leg(market)
        return receiver if self.direction == "receive_equity" else -receiver

    def describe(self):
        side = ("receive equity / PAY funding (short funding leg)"
                if self.direction == "receive_equity" else "pay equity / receive funding")
        u = (self.positions[0].isin if len(self.positions) == 1
             else f"basket of {len(self.positions)} names")
        return (f"Equity TRS on {u}: {side}, {len(self.positions)} position(s), "
                f"{self.trade_currency}, spread {self.spread*1e4:.0f}bp, "
                f"{self.start_date} -> {self.end_date}")
