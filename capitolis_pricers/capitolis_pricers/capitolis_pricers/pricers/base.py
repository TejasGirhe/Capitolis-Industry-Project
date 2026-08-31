"""Pricer base class -- the repricing contract every product honours."""

from ..market import MarketState


class Pricer:
    trade_currency = "USD"
    counterparty = None      # optional tag for netting-set / CVA grouping

    def _npv_trade_ccy(self, market: MarketState) -> float:
        raise NotImplementedError

    def npv(self, market: MarketState, reporting: bool = False) -> float:
        """Price on `market`; if reporting=True convert to market.reporting_ccy."""
        value = self._npv_trade_ccy(market)
        if reporting and self.trade_currency.upper() != market.reporting_ccy.upper():
            value *= market.fx(self.trade_currency, market.reporting_ccy)
        return value

    def describe(self) -> str:
        raise NotImplementedError
