"""
Load the underlyings (in trade_data/underlyings/).

equities.csv is the per-trade basket + identity, keyed by trade_id (equities map
DOWN from a trade to its basket). bonds.csv holds bond definitions keyed by
bond_id (bonds map UP from one bond to its trade). Identifier columns
(ticker/isin/cusip) are for mapping and are not used in pricing.
"""
import csv


def load_equities(path):
    """{trade_id: [{isin, shares, basis, currency}, ...]} -- the basket.
    ISIN is the security identifier that keys the market data (spot, dividend)."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            b = (r.get("basis") or "").strip()
            out.setdefault(r["trade_id"].strip(), []).append(dict(
                isin=r["isin"].strip(),
                shares=float(r["shares"]),
                basis=None if b == "" else float(b),
                currency=r["currency"].strip()))
    return out


def load_bonds(path):
    """{bond_id: FixedRateBond kwargs (+ currency)}."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["bond_id"].strip()] = dict(
                issue_date=r["issue_date"].strip(),
                maturity_date=r["maturity_date"].strip(),
                coupon=float(r["coupon"]),
                frequency_months=int(r["coupon_frequency_months"]),
                day_count=r["day_count"].strip(),
                face=float(r.get("par", 100) or 100),
                issuer=(r.get("issuer") or "").strip() or None,
                currency=r.get("currency", "USD").strip())
    return out
