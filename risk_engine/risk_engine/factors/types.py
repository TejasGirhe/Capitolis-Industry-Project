"""
Risk factor identity types.

Each factor is a frozen, hashable key -- used to index correlation matrices
and calibrated-model caches. Factors carry only identity (what market
quantity they refer to), never levels or model choice; extraction is a pure
function of the book, independent of which stochastic model will later drive
each factor.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class RateFactor:
    """A discount/short-rate curve risk factor, e.g. RateFactor('USD')."""
    currency: str

    def __str__(self):
        return f"RATE_{self.currency}"


@dataclass(frozen=True)
class EquityFactor:
    """A single-name equity spot risk factor, keyed by ISIN (market-data key)."""
    isin: str
    native_ccy: str  # the name's own quote currency; != trade_ccy => compo

    def __str__(self):
        return f"EQ_{self.isin}"


@dataclass(frozen=True)
class FxFactor:
    """An FX risk factor for a compo trade, quoted quote-per-base (e.g. base=USD, quote=JPY -> USDJPY)."""
    base_ccy: str
    quote_ccy: str

    def __str__(self):
        return f"FX_{self.base_ccy}{self.quote_ccy}"


@dataclass(frozen=True)
class CreditFactor:
    """Issuer credit curve factor -- only active under the optional risky-bond extension."""
    issuer: str

    def __str__(self):
        return f"CREDIT_{self.issuer}"


@dataclass
class FactorSet:
    """The distinct risk factors driving a book, as extracted from its trades."""
    rates: Tuple[RateFactor, ...] = field(default_factory=tuple)
    equities: Tuple[EquityFactor, ...] = field(default_factory=tuple)
    fx: Tuple[FxFactor, ...] = field(default_factory=tuple)
    credit: Tuple[CreditFactor, ...] = field(default_factory=tuple)

    def all(self):
        """Flat, stably-ordered list of every factor -- for correlation-matrix indexing."""
        return [*self.rates, *self.equities, *self.fx, *self.credit]

    def __len__(self):
        return len(self.all())
