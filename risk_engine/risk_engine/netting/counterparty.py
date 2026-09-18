"""
Counterparty: aggregates its NettingSet(s) into one collateralized exposure
number per (path, date, curve) node. Currently 1:1 with NettingSet per
counterparty (build_netting_hierarchy assigns exactly one DUMMY netting-
agreement id per counterparty -- see netting_set.py's module docstring),
but Counterparty.netting_sets is already a list specifically so a
counterparty with multiple real netting agreements is a config change to
build_netting_hierarchy's grouping key, not a redesign of this class or of
any exposure code that iterates netting_sets.
"""
from dataclasses import dataclass
from typing import Dict, List

from .netting_set import NettingSet

DUMMY_NETTING_AGREEMENT_SUFFIX = "-ISDA-DUMMY-01"


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
    Counterparty per counterparty id.

    Each NettingSet gets a DUMMY netting-agreement id
    (f"{cpty}{DUMMY_NETTING_AGREEMENT_SUFFIX}"), NOT the bare counterparty
    id -- this project has no real ISDA/CSA agreement feed (no per-
    counterparty agreement count or product-eligibility data), so there is
    exactly one agreement per counterparty here, same as before. What
    changed is that the netting-set id is no longer *structurally*
    identical to the counterparty id: a real agreement feed can replace
    this function's grouping key (e.g. group by trade.isda_agreement_id
    instead of trade.counterparty, one NettingSet per real agreement, still
    nested under the same counterparty) without any change to NettingSet,
    Counterparty, or the exposure/pricing code that consumes them."""
    by_cpty: Dict[str, List[str]] = {}
    for tid, trade in trades.items():
        cpty = trade.counterparty
        if cpty is None:
            raise ValueError(f"trade '{tid}' has no counterparty set -- cannot build netting hierarchy")
        by_cpty.setdefault(cpty, []).append(tid)

    return [
        Counterparty(id=cpty, netting_sets=[
            NettingSet(id=f"{cpty}{DUMMY_NETTING_AGREEMENT_SUFFIX}", counterparty_id=cpty, trade_ids=sorted(tids))
        ])
        for cpty, tids in sorted(by_cpty.items())
    ]
