"""
Price the sample trades. Capitolis inputs load from trade_data/; the market is
built in code from sample values (students build theirs from MARKET_DATA.md).
Default direction: CLGM sells the TRS (pays total return, receives funding).

    python -m capitolis_pricers.examples.price_all
"""
import os
from capitolis_pricers.data.sample_market import sample_market
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import (
    load_equity_trs, load_bond_forward, load_bond_trs)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
TD = os.path.join(ROOT, "trade_data"); U = os.path.join(TD, "underlyings")


def main():
    market = sample_market()
    baskets = load_equities(os.path.join(U, "equities.csv"))
    bonds = load_bonds(os.path.join(U, "bonds.csv"))

    trades = {}
    trades.update(load_equity_trs(os.path.join(TD, "equity_trs.csv"), baskets))
    trades.update(load_bond_forward(os.path.join(TD, "bond_forward.csv"), bonds))
    trades.update(load_bond_trs(os.path.join(TD, "bond_trs.csv"), bonds))

    print(f"Valuation {market.ref_date}   reporting {market.reporting_ccy}   "
          f"(CLGM sells the TRS: pays total return, receives funding)\n")
    for tid, p in trades.items():
        print(f"{tid:12s} NPV = {p.npv(market, reporting=True):15,.2f} {market.reporting_ccy}")
        print(f"{'':12s} {p.describe()}\n")


if __name__ == "__main__":
    main()
