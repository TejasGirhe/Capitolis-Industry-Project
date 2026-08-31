"""
SA-CCR delta (Basel III Standardized Approach for Counterparty Credit Risk,
Annex 4) -- the regulatory-capital delta input Slide 15 ties to
EAD = alpha * (RC + PFE_addon). This is a CLOSED-FORM formula per Basel's
own supervisory conventions, not a shocked-and-repriced sensitivity: no
simulation is involved, and it is NOT the same thing as "vega" (see
greeks/vega.py for the bump-and-reprice sensitivity, which SA-CCR itself
has no equivalent of -- its PFE multiplier uses a fixed supervisory
volatility factor per asset class, not a computed one).

Supervisory duration (Annex 4 para 5):
    SD(S,E) = (exp(-0.05*S) - exp(-0.05*E)) / 0.05
applies to interest-rate and credit trades (time-value discounting of a
notional cashflow stream). FX and equity are linear with supervisory delta
= 1 for the whole trade (no duration weighting -- Annex 4 para 4/6/8:
"delta = +1 or -1" for non-option FX/equity trades), so their SA-CCR
"delta" is just signed adjusted notional.

Supervisory factors (Annex 4 Table 2, current Basel III standardized
values): interest rate 0.50%, FX 4.0%, equity single-name 32%, equity
index 20%. These feed the AddOn calculation
(AddOn = supervisory_factor * |sum(delta * adjusted_notional)|
per hedging set) -- this module computes the delta*notional term (the
"effective notional"), not the full multi-hedging-set AddOn/EAD chain,
which is a substantially larger regulatory-capital aggregation left out of
scope (see aggregate_delta's docstring).
"""
from typing import Dict

from .sensitivities import notional_of, direction_sign, asset_class_of, trade_start_end

SUPERVISORY_FACTOR = {
    "interest_rate": 0.0050,
    "foreign_exchange": 0.0400,
    "equity_single_name": 0.32,
    "equity_index": 0.20,
}


def supervisory_duration(S: float, E: float) -> float:
    """SD(S,E) = (exp(-0.05*S) - exp(-0.05*E)) / 0.05 -- Basel III Annex 4 para 5."""
    import math
    return (math.exp(-0.05 * S) - math.exp(-0.05 * E)) / 0.05


def rate_delta(trade, ref_date) -> float:
    """Interest-rate asset class: notional x SD(S,E) x sign x supervisory_factor
    -- the AddOn-feeding "effective notional" for a rate trade (bond forward
    / bond TRS -- both risk-free, USD-curve-only per README Sec.7/8)."""
    S, E = trade_start_end(trade, ref_date)
    sd = supervisory_duration(S, E)
    return notional_of(trade) * sd * direction_sign(trade) * SUPERVISORY_FACTOR["interest_rate"]


def fx_delta(trade, ref_date) -> float:
    """FX asset class: adjusted_notional x sign x supervisory_factor. Linear
    FX has supervisory delta = 1 (no duration weighting, no bump) per Annex
    4 para 4."""
    return notional_of(trade) * direction_sign(trade) * SUPERVISORY_FACTOR["foreign_exchange"]


def equity_delta(trade, ref_date, index: bool = False) -> float:
    """Equity asset class: adjusted_notional x sign x supervisory_factor.
    `index` selects the single-name (32%) vs index (20%) supervisory
    factor (Annex 4 Table 2) -- every equity name in this book is a
    single-name basket constituent (equities.csv's basket rows are
    individual names, not an index proxy), so this defaults to
    single-name; pass index=True if a trade's underlying is confirmed to
    be an index product."""
    factor_key = "equity_index" if index else "equity_single_name"
    return notional_of(trade) * direction_sign(trade) * SUPERVISORY_FACTOR[factor_key]


def trade_delta(trade, ref_date) -> Dict[str, float]:
    """{asset_class: delta} for one trade -- a compo equity TRS (FX +
    equity legs both live, see sensitivities.is_compo) gets BOTH an
    'equity' and a 'foreign_exchange' key; every other trade gets exactly
    one key.

    Credit spread delta is intentionally NOT a routable asset class here:
    every bond in this book is risk-free (no issuer CreditCurve populated
    in MarketState.credit_curves by default -- see
    factors.extract.extract_factors's include_risky_bond_credit flag,
    off by default), so there is no CDS/credit-spread factor to be
    sensitive to. If the risky-bond extension is ever wired in, a
    'credit_spread_delta' formula and asset class belong here -- flagged
    as the scoping boundary, not silently mishandled.
    """
    ac = asset_class_of(trade)
    if ac == "interest_rate":
        return {"interest_rate": rate_delta(trade, ref_date)}
    if ac == "equity":
        return {"equity": equity_delta(trade, ref_date)}
    if ac == "equity_fx_compo":
        return {"equity": equity_delta(trade, ref_date), "foreign_exchange": fx_delta(trade, ref_date)}
    raise ValueError(f"unrouted asset class '{ac}'")


def aggregate_delta(trades: Dict[str, object], ref_date) -> Dict[str, Dict[str, float]]:
    """{counterparty: {asset_class: summed_delta}} -- the delta INPUT to
    SA-CCR's AddOn = supervisory_factor * |sum(delta)| per hedging set
    (Slide 15's EAD = alpha * (RC + PFE_addon) framing). This function
    stops at the per-counterparty, per-asset-class summed delta; it does
    NOT build the full multi-hedging-set maturity-bucket AddOn aggregation
    or the RC/PFE_addon/EAD chain itself -- that is a substantially larger
    regulatory-capital module (hedging-set correlation parameters,
    maturity bucketing within interest-rate, multiplier formula, etc.)
    intentionally out of scope for this pass. Stated as a scoping
    boundary, not a silent gap.
    """
    out: Dict[str, Dict[str, float]] = {}
    for trade in trades.values():
        cpty = trade.counterparty
        deltas = trade_delta(trade, ref_date)
        bucket = out.setdefault(cpty, {})
        for asset_class, delta in deltas.items():
            bucket[asset_class] = bucket.get(asset_class, 0.0) + delta
    return out
