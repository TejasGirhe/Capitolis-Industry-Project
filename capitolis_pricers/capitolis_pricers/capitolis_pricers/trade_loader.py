"""
Load trades (in trade_data/).

    from capitolis_pricers.underlyings_loader import load_equities, load_bonds
    from capitolis_pricers.trade_loader import (
        load_equity_trs, load_bond_forward, load_bond_trs)

    baskets = load_equities("trade_data/underlyings/equities.csv")
    bonds   = load_bonds("trade_data/underlyings/bonds.csv")
    eqtrs   = load_equity_trs("trade_data/equity_trs.csv", baskets)
    bfwd    = load_bond_forward("trade_data/bond_forward.csv", bonds)
    btrs    = load_bond_trs("trade_data/bond_trs.csv", bonds)
"""
import csv
from .pricers import EquityTRS, BondForwardTrade, BondTRS


def _f(row, key, default=None):
    v = row.get(key, "")
    v = v.strip() if isinstance(v, str) else v
    return default if v in ("", None) else float(v)


def load_equity_trs(path, baskets):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            tid = r["trade_id"].strip()
            if tid not in baskets:
                raise KeyError(f"no basket rows for trade '{tid}' in equities.csv")
            out[tid] = EquityTRS(
                positions=baskets[tid],
                trade_ccy=r["trade_ccy"].strip(),
                start_date=r["start_date"].strip(),
                end_date=r["end_date"].strip(),
                reset_frequency_months=int(float(r.get("reset_frequency_months") or 0)),
                spread=_f(r, "spread", 0.0),
                direction=r["direction"].strip(),
                fixed_rate=_f(r, "fixed_rate", None),
                funding_notional=_f(r, "funding_notional", None),
                funding_index=(r.get("funding_index") or "").strip() or None)
            out[tid].counterparty = (r.get("counterparty") or "").strip() or None
    return out


def _terms(bonds, bond_id):
    if bond_id not in bonds:
        raise KeyError(f"bond '{bond_id}' not in bonds.csv")
    return {k: v for k, v in bonds[bond_id].items() if k != "currency"}


def load_bond_forward(path, bonds):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["trade_id"].strip()] = BondForwardTrade.from_terms(
                bond_terms=_terms(bonds, r["bond_id"].strip()),
                forward_date=r["forward_date"].strip(),
                strike_clean=float(r["strike_clean"]),
                position=r["position"].strip(),
                trade_ccy=r["trade_ccy"].strip(),
                notional=float(r["notional"]),
                repo_rate=_f(r, "repo_rate", None))
            out[r["trade_id"].strip()].counterparty = (r.get("counterparty") or "").strip() or None
    return out


def load_bond_trs(path, bonds):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["trade_id"].strip()] = BondTRS.from_terms(
                bond_terms=_terms(bonds, r["bond_id"].strip()),
                notional=float(r["notional"]),
                start_date=r["start_date"].strip(),
                end_date=r["end_date"].strip(),
                reset_frequency_months=int(float(r.get("reset_frequency_months") or 0)),
                spread=_f(r, "spread", 0.0),
                fixed_rate=_f(r, "fixed_rate", None),
                funding_notional=_f(r, "funding_notional", None),
                direction=r["direction"].strip(),
                trade_ccy=r["trade_ccy"].strip())
            out[r["trade_id"].strip()].counterparty = (r.get("counterparty") or "").strip() or None
    return out
