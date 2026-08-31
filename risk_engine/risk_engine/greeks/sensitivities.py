"""
Generic trade adapters -- EquityTRS, BondForwardTrade, and BondTRS each name
their notional, side, and maturity attributes differently (confirmed by
direct inspection: EquityTRS/BondTRS use `.direction`
(pay_equity/receive_equity, pay_tr/receive_tr) and EquityTRS has no plain
`.notional`, only a nullable `.funding_notional`; BondForwardTrade uses
`.position` (long/short) and a required `.notional`). risk_engine.pricing.
trade_maturity() and risk_engine.simulation.grid.trade_cashflow_dates()
already duck-type across this split for maturity/cashflow-date purposes;
this module adds the same pattern for notional/side/asset-class, needed by
the SA-CCR delta formulas in sa_ccr.py.
"""
from datetime import date

from capitolis_pricers.pricers.equity_trs import EquityTRS
from capitolis_pricers.pricers.bond_forward import BondForwardTrade
from capitolis_pricers.pricers.bond_trs import BondTRS

from ..pricing import trade_maturity


def notional_of(trade) -> float:
    """The SA-CCR 'adjusted notional' base -- before any supervisory
    duration/delta multiplier is applied."""
    if hasattr(trade, "notional"):                      # BondForwardTrade, BondTRS
        return trade.notional
    if hasattr(trade, "funding_notional"):               # EquityTRS
        if trade.funding_notional is not None:
            return trade.funding_notional
        return sum(p.shares * (p.basis or 0.0) for p in trade.positions)
    raise AttributeError(f"{type(trade).__name__} has no notional/funding_notional")


def direction_sign(trade) -> int:
    """+1 = long the underlying's return (receiver of equity/total return,
    or long the bond forward), -1 = short (payer) -- SA-CCR's supervisory
    delta sign convention (Basel III Annex 4 para 4: delta is +1 for a
    long position, -1 for short, before the supervisory-delta-for-options
    adjustment, which doesn't apply to these linear products)."""
    if hasattr(trade, "position"):                       # BondForwardTrade: long/short
        return 1 if trade.position == "long" else -1
    if trade.direction in ("receive_equity", "receive_tr"):
        return 1
    if trade.direction in ("pay_equity", "pay_tr"):
        return -1
    raise ValueError(f"unrecognized direction '{trade.direction}' on {type(trade).__name__}")


def is_compo(trade) -> bool:
    """True if any constituent's native currency differs from the trade's
    settlement currency -- i.e. the trade also carries FX risk (compo)."""
    if not isinstance(trade, EquityTRS):
        return False
    return any(p.currency != trade.trade_currency for p in trade.positions)


def asset_class_of(trade) -> str:
    """SA-CCR asset class bucket: 'interest_rate' (bonds -- risk-free,
    discounted on the USD curve only, per capitolis_pricers README Sec.7/8),
    'equity' (equity TRS), or 'equity_fx_compo' (equity TRS with a
    currency-mismatched leg -- carries BOTH equity and FX delta; see
    sa_ccr.trade_delta, which returns both keys for this class)."""
    if isinstance(trade, (BondForwardTrade, BondTRS)):
        return "interest_rate"
    if isinstance(trade, EquityTRS):
        return "equity_fx_compo" if is_compo(trade) else "equity"
    raise TypeError(f"unrecognized trade type {type(trade).__name__}")


def trade_start_end(trade, ref_date) -> tuple:
    """(S, E) in years from ref_date, for SA-CCR's supervisory duration --
    S=0 if the trade's own start predates ref_date (Basel's supervisory
    duration measures time remaining FROM TODAY, not from the trade's
    original inception; a trade already live has S=0, not a negative
    start), matching the maturity guard's own "as of ref_date" framing
    already used by pricing.price_curves.

    E is the trade's own maturity (trade_maturity() -- end_date for
    EquityTRS/BondTRS, forward_date for BondForwardTrade).
    """
    maturity = trade_maturity(trade)
    E = max((maturity - ref_date).days / 365.0, 0.0)
    start = getattr(trade, "start_date", ref_date)  # BondForwardTrade has no start_date -- trade exists from ref_date
    S = max((start - ref_date).days / 365.0, 0.0)
    return S, E
