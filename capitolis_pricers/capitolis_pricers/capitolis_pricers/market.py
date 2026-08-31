"""
MarketState -- market data only (levels), the pricers' repricing input.

Market data is what the students COLLECT: discount curves, equity spots, FX
spots (and, for their engine, vols and correlations -- held here, not used to
price). Static instrument attributes (currency, dividend rate, bond terms) live
in the underlyings layer and are baked into the trade objects, NOT here.

    npv = pricer.npv(market)                  # NPV in the trade's currency
    npv = pricer.npv(market, reporting=True)  # NPV in market.reporting_ccy
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Tuple

from .curves import Curve, FxCurve
from .daycount import to_date


@dataclass
class MarketState:
    ref_date: date
    reporting_ccy: str = "USD"
    discount_curves: Dict[str, Curve] = field(default_factory=dict)
    equity_spots: Dict[str, float] = field(default_factory=dict)     # {ISIN: spot}
    equity_dividend_rates: Dict[str, float] = field(default_factory=dict)  # {ISIN: q}
    fx_curves: Dict[Tuple[str, str], FxCurve] = field(default_factory=dict)
    # held for the students' engine; NOT used to price the linear products:
    equity_vols: Dict[str, float] = field(default_factory=dict)
    correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    credit_curves: Dict[str, object] = field(default_factory=dict)   # {counterparty: CreditCurve} (optional, for CVA)

    def __post_init__(self):
        self.ref_date = to_date(self.ref_date)

    def discount(self, ccy):
        ccy = ccy.upper()
        if ccy not in self.discount_curves:
            raise KeyError(f"No discount curve for {ccy}")
        return self.discount_curves[ccy]

    def equity_spot(self, isin):
        """Spot looked up by the equity's security identifier (ISIN)."""
        if isin not in self.equity_spots:
            raise KeyError(f"No spot for security '{isin}'")
        return self.equity_spots[isin]

    def equity_dividend(self, isin):
        """Dividend rate (continuous yield), market data keyed by ISIN.
        Students source it however they like (implied, put-call parity, ...)."""
        return self.equity_dividend_rates.get(isin, 0.0)

    def fx(self, from_ccy, to_ccy):
        from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()
        if from_ccy == to_ccy:
            return 1.0
        for (b, q), fxc in self.fx_curves.items():
            if (from_ccy, to_ccy) == (b, q):
                return fxc.spot
            if (from_ccy, to_ccy) == (q, b):
                return 1.0 / fxc.spot
        raise KeyError(f"No FX for {from_ccy}->{to_ccy}")

    def credit(self, counterparty):
        """Optional counterparty CreditCurve for CVA (None if not supplied)."""
        return self.credit_curves.get(counterparty)

    def correlation(self, a, b):
        if a == b:
            return 1.0
        return self.correlations.get((a, b)) or self.correlations.get((b, a)) or 0.0
