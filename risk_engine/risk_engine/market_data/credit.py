"""
Counterparty credit spread proxy -- for CVA/DVA. No free source publishes
name-specific CDS curves (confirmed by direct investigation: this book's
CPTY_A/B/C are not real entities anyway, and no free data source exists for
real single-name CDS spreads either -- that is a Markit/Bloomberg paid
product, same conclusion reached earlier for the rate vol surface).

What IS real and free: FRED's Moody's Seasoned Aaa/Baa corporate bond yield
series, minus the 10Y Treasury yield, gives a genuine RATING-TIER credit
spread (confirmed via direct fetch: AAA ~5.76%, BAA ~6.19%, DGS10 ~4.69% ->
AAA spread ~107bp, BAA spread ~150bp over risk-free). This is a rating-BUCKET
proxy, not a counterparty-specific curve -- the same honest-substitution
pattern already used elsewhere in this package (Treasuries standing in for
SOFR OIS in rates.py; CIP-derived, not observed, FX forwards in fx.py).

Confirmed tiering design: counterparties are ranked by total book size
(sum of absolute notional across their trades) and mapped to a credit tier --
largest book -> AAA proxy, smallest -> BAA proxy, everything in between
interpolated linearly between the two spreads. This is a stated assumption
(larger relationship assumed more institutional/creditworthy), not a real
credit assessment.

Where this plugs in: capitolis_pricers.credit.CreditCurve (already exists in
the provided pricing library, built for exactly this -- survival probability
via the credit-triangle approximation h=s/(1-R), S(t)=exp(-h*t)) is
instantiated directly here, flat across tenors (no term structure available
from this data), and returned ready to drop into
capitolis_pricers.market.MarketState.credit_curves (already counterparty-
keyed, per that field's own inline comment: "{counterparty: CreditCurve}
(optional, for CVA)").
"""
from typing import Dict

from capitolis_pricers.credit import CreditCurve

from ._fred import fetch_latest_fred_value
from ..greeks.sensitivities import notional_of

FRED_AAA_SERIES = "AAA"      # Moody's Seasoned Aaa Corporate Bond Yield
FRED_BAA_SERIES = "BAA"      # Moody's Seasoned Baa Corporate Bond Yield
FRED_UST10_SERIES = "DGS10"  # already used by rates.py's USD_CURVE_SERIES

CREDIT_TENORS_YEARS = (0.5, 1.0, 2.0, 5.0, 10.0)  # matches the book's own USD curve tenor convention


def fetch_rating_tier_spread(tier: str) -> float:
    """AAA or BAA spread over the 10Y Treasury, from real FRED data.
    Documented proxy: a rating-BUCKET spread, not a name-specific CDS curve."""
    tier = tier.upper()
    series = {"AAA": FRED_AAA_SERIES, "BAA": FRED_BAA_SERIES}.get(tier)
    if series is None:
        raise ValueError(f"unknown rating tier '{tier}', expected 'AAA' or 'BAA'")
    corp_yield = fetch_latest_fred_value(series)
    ust10 = fetch_latest_fred_value(FRED_UST10_SERIES)
    return corp_yield - ust10


def book_size_by_counterparty(trades: Dict[str, object]) -> Dict[str, float]:
    """{counterparty: total absolute notional across its trades} -- the
    ranking input for tier_for_counterparty. Reuses greeks.sensitivities.
    notional_of, the same trade-notional adapter built for SA-CCR delta,
    rather than re-deriving notional extraction here."""
    out: Dict[str, float] = {}
    for trade in trades.values():
        cpty = trade.counterparty
        out[cpty] = out.get(cpty, 0.0) + abs(notional_of(trade))
    return out


def spread_for_counterparty(trades: Dict[str, object], counterparty: str,
                             aaa_spread: float, baa_spread: float) -> float:
    """Linear interpolation between the AAA spread (largest book) and BAA
    spread (smallest book), by this counterparty's rank in book size among
    all counterparties in `trades`. A book with only one counterparty gets
    the AAA (best) tier; two counterparties get AAA/BAA exactly; three or
    more interpolate linearly by rank."""
    sizes = book_size_by_counterparty(trades)
    if counterparty not in sizes:
        raise KeyError(f"no trades found for counterparty '{counterparty}'")
    ranked = sorted(sizes, key=lambda c: sizes[c], reverse=True)  # largest book first
    n = len(ranked)
    if n == 1:
        return aaa_spread
    rank = ranked.index(counterparty)  # 0 = largest book (best tier)
    w = rank / (n - 1)  # 0.0 at largest book -> AAA, 1.0 at smallest book -> BAA
    return aaa_spread + w * (baa_spread - aaa_spread)


def counterparty_spreads(trades: Dict[str, object]) -> Dict[str, float]:
    """{counterparty: flat spread} -- the same tiering CreditCurve.build_credit_curves
    uses internally, exposed standalone for reporting (CreditCurve itself has
    no public accessor for the spread it was built from -- _spread()/_s are
    private, so this is the intended way to recover "what spread did this
    counterparty get" without reaching into CreditCurve's internals)."""
    aaa_spread = fetch_rating_tier_spread("AAA")
    baa_spread = fetch_rating_tier_spread("BAA")
    counterparties = sorted({t.counterparty for t in trades.values()})
    return {cpty: spread_for_counterparty(trades, cpty, aaa_spread, baa_spread) for cpty in counterparties}


def build_credit_curves(trades: Dict[str, object], ref_date, recovery: float = 0.40) -> Dict[str, CreditCurve]:
    """{counterparty: CreditCurve} -- one per counterparty in `trades`, FLAT
    spread across CREDIT_TENORS_YEARS (no term structure available from this
    data source -- same flat-proxy pattern as market_data.jpy_rate's
    flat_jpy_rate_by_tenor). recovery=0.40 is capitolis_pricers.credit.
    CreditCurve's own documented default, not re-derived here.
    """
    aaa_spread = fetch_rating_tier_spread("AAA")
    baa_spread = fetch_rating_tier_spread("BAA")

    counterparties = sorted({t.counterparty for t in trades.values()})
    out = {}
    for cpty in counterparties:
        spread = spread_for_counterparty(trades, cpty, aaa_spread, baa_spread)
        out[cpty] = CreditCurve(ref_date, CREDIT_TENORS_YEARS, [spread] * len(CREDIT_TENORS_YEARS), recovery=recovery)
    return out


def build_own_credit_curve(ref_date, recovery: float = 0.40) -> CreditCurve:
    """Capitolis's own credit curve, for DVA's own-default leg and (as a
    funding-spread proxy) FVA. Reuses the SAME BAA-tier spread as a
    conservative default (a mid-tier, unrated-but-active market
    participant) -- flagged as a placeholder a real desk would replace with
    its own actual funding/CDS curve, not a real Capitolis-specific number."""
    baa_spread = fetch_rating_tier_spread("BAA")
    return CreditCurve(ref_date, CREDIT_TENORS_YEARS, [baa_spread] * len(CREDIT_TENORS_YEARS), recovery=recovery)
