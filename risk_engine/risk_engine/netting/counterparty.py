"""
Counterparty: aggregates its NettingSet(s) into one collateralized exposure
number per (path, date, curve) node. 1:1 with NettingSet for this book
(single netting agreement per counterparty), kept as its own aggregation
step so a counterparty with multiple netting sets is a config change, not a
redesign.
"""
from dataclasses import dataclass
from typing import Dict, List

from .netting_set import NettingSet


@dataclass
class Counterparty:
    id: str
    netting_sets: List[NettingSet]

    def collateralized_exposure(self, curve_result, curve: str, margin_model, path: int, date,
                                 exclude_trade_ids=()) -> float:
        return sum(
            ns.netted_npv(curve_result, curve, path, date, exclude_trade_ids)
            - margin_model.posted_collateral(ns, curve_result, curve, path, date)
            for ns in self.netting_sets
        )


def build_netting_hierarchy(trades: Dict[str, object]) -> List[Counterparty]:
    """Group trades by their .counterparty attribute (set by trade_loader.py
    on every loaded Pricer) into one NettingSet per counterparty, one
    Counterparty per counterparty id."""
    by_cpty: Dict[str, List[str]] = {}
    for tid, trade in trades.items():
        cpty = trade.counterparty
        if cpty is None:
            raise ValueError(f"trade '{tid}' has no counterparty set -- cannot build netting hierarchy")
        by_cpty.setdefault(cpty, []).append(tid)

    return [
        Counterparty(id=cpty, netting_sets=[NettingSet(id=cpty, trade_ids=sorted(tids))])
        for cpty, tids in sorted(by_cpty.items())
    ]
