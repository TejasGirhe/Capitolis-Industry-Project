"""
capitolis_pricers -- standalone pricing models (no third-party dependency).

Standalone re-implementation of the discounted-cash-flow methodology used by
QuantLib / ORE (the Open Source Risk Engine). Scope: pricing models + their
inputs. The Monte Carlo / exposure engine is the students' deliverable.

Data layers: underlyings (static, Capitolis) + trades (Capitolis) + market data
(collected by students). See README.md.
"""
from .daycount import to_date
from .curves import Curve, flat_curve, zero_curve, FxCurve
from .market import MarketState
from .bond import FixedRateBond
from .pricers import EquityTRS, BondForwardTrade, BondTRS
from .credit import CreditCurve
from . import underlyings_loader, trade_loader

__all__ = ["to_date", "Curve", "flat_curve", "zero_curve", "FxCurve", "MarketState",
           "FixedRateBond", "EquityTRS", "BondForwardTrade", "BondTRS", "CreditCurve",
           "underlyings_loader", "trade_loader"]
__version__ = "0.3.0"
