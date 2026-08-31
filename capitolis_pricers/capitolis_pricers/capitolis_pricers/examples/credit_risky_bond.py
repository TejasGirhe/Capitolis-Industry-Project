"""
OPTIONAL: risky-bond (issuer credit) demo.

Bonds ship risk-free. To extend to RISKY bonds, supply the issuer's CreditCurve
in MarketState.credit_curves (keyed by the bond's `issuer`); the bond then prices
credit-risky, and its credit charge ("CVA") is:

    credit charge = risk-free price - credit-risky price

    python -m capitolis_pricers.examples.credit_risky_bond
"""
import copy
from capitolis_pricers import CreditCurve, BondForwardTrade, FixedRateBond
from capitolis_pricers.data.sample_market import sample_market

TERMS = dict(issue_date="2024-01-15", maturity_date="2031-01-15", coupon=0.04,
             frequency_months=6, day_count="30/360", face=100, issuer="ACME_CORP")


def main():
    market = sample_market()
    bond = FixedRateBond(**TERMS)
    usd = market.discount("USD")

    rf = bond.dirty_price(usd, market.ref_date)                  # risk-free

    # attach an issuer credit curve (this is the optional extension)
    m = copy.copy(market); m.credit_curves = dict(market.credit_curves)
    m.credit_curves["ACME_CORP"] = CreditCurve(
        market.ref_date, [1, 3, 5, 10], [0.0100, 0.0150, 0.0200, 0.0250], recovery=0.40)
    risky = bond.dirty_price(usd, market.ref_date, m.credit("ACME_CORP"))

    print(f"Bond {TERMS['coupon']*100:.2f}% {TERMS['maturity_date']}  issuer {TERMS['issuer']}")
    print(f"  risk-free dirty price : {rf:.4f}")
    print(f"  credit-risky price    : {risky:.4f}")
    print(f"  credit charge (CVA)   : {rf - risky:.4f}  (per 100 face)\n")

    # flows straight through a bond forward on the same (now risky) issuer
    bf = BondForwardTrade(bond, forward_date="2026-07-15", strike_clean=98.50,
                          position="long", trade_ccy="USD", notional=10_000_000)
    print(f"  bond forward NPV risk-free: {bf.npv(market):15,.2f} USD")
    print(f"  bond forward NPV risky    : {bf.npv(m):15,.2f} USD")


if __name__ == "__main__":
    main()
