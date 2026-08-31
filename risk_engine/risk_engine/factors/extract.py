"""
Book-level risk factor extraction.

Walks every trade in the book and derives the distinct set of risk factors
that must be simulated -- independent of which stochastic model will later
drive each factor. Pure function of the loaded trade/bond objects from
capitolis_pricers.trade_loader / underlyings_loader; no market data or model
dependency.

    from capitolis_pricers.underlyings_loader import load_equities, load_bonds
    from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs

    baskets = load_equities("trade_data/underlyings/equities.csv")
    bonds   = load_bonds("trade_data/underlyings/bonds.csv")
    eqtrs   = load_equity_trs("trade_data/equity_trs.csv", baskets)
    bfwd    = load_bond_forward("trade_data/bond_forward.csv", bonds)
    btrs    = load_bond_trs("trade_data/bond_trs.csv", bonds)

    factors = extract_factors(eqtrs, bfwd, btrs)
"""
from .types import RateFactor, EquityFactor, FxFactor, CreditFactor, FactorSet


def extract_factors(equity_trs=None, bond_forwards=None, bond_trs=None,
                     include_risky_bond_credit=False):
    """Derive the FactorSet driving a book.

    equity_trs, bond_forwards, bond_trs: {trade_id: Pricer} dicts as returned
        by capitolis_pricers.trade_loader. Each BondForwardTrade/BondTRS holds
        its bond as `.bond` (a FixedRateBond with an `.issuer` attribute).
    include_risky_bond_credit: if True, bond trades whose bond has a non-blank
        `.issuer` also contribute a CreditFactor(issuer). Off by default:
        bonds ship risk-free per README Sec.7/8 and MARKET_DATA.md Sec.4 treats
        issuer credit as an optional extension, not a default risk factor.

    Any USD-discounted trade is present (equity TRS, bond forward, bond TRS
    all discount/fund off the USD curve per README Sec.7), so a single
    RateFactor('USD') is added whenever the book is non-empty.

    On the current book this returns 1 RateFactor, 37 EquityFactor (41 basket
    rows, 4 ISINs repeated across trades), and 1 FxFactor(USD, JPY) -- two
    trades (EQTRS_0005, EQTRS_0006) hold JPY-currency basket rows against a
    USD trade_ccy, i.e. compo.
    """
    equity_trs = equity_trs or {}
    bond_forwards = bond_forwards or {}
    bond_trs = bond_trs or {}

    currencies = set()
    equities = {}   # isin -> EquityFactor, dedup by isin
    fx_pairs = set()

    for trade in equity_trs.values():
        currencies.add(trade.trade_currency)
        for pos in trade.positions:
            if pos.isin not in equities:
                equities[pos.isin] = EquityFactor(isin=pos.isin, native_ccy=pos.currency)
            if pos.currency != trade.trade_currency:
                fx_pairs.add((trade.trade_currency, pos.currency))

    credit_issuers = set()
    for trade in list(bond_forwards.values()) + list(bond_trs.values()):
        currencies.add(trade.trade_currency)
        if include_risky_bond_credit:
            issuer = getattr(trade.bond, "issuer", None)
            if issuer:
                credit_issuers.add(issuer)

    if not currencies and (equity_trs or bond_forwards or bond_trs):
        currencies.add("USD")

    rates = tuple(sorted((RateFactor(c) for c in currencies), key=lambda r: r.currency))
    eq = tuple(sorted(equities.values(), key=lambda e: e.isin))
    fx = tuple(sorted((FxFactor(b, q) for b, q in fx_pairs), key=lambda f: (f.base_ccy, f.quote_ccy)))
    credit = tuple(sorted((CreditFactor(i) for i in credit_issuers), key=lambda c: c.issuer))

    return FactorSet(rates=rates, equities=eq, fx=fx, credit=credit)
