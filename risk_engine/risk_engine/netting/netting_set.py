"""
NettingSet: a group of trades netted together for exposure purposes. For
this book, netting_set.id == counterparty (capitolis_pricers' README:
"every trade carries a counterparty -- the netting-set key your simulation
groups by to net exposures"), one NettingSet per counterparty. Kept as its
own class rather than collapsed into Counterparty so a book with multiple
netting agreements per counterparty is a config change later, not a
redesign.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class NettingSet:
    id: str
    trade_ids: List[str]

    def netted_npv(self, curve_result, curve: str, path: int, date, exclude_trade_ids=()) -> float:
        """curve: 'npv0' or 'npv10' -- risk_engine.pricing.CurveResult's two
        curves (value AT date, and the MPoR curve value at date+mpor_days).
        exclude_trade_ids: trades to leave out of this sum -- used to keep a
        close-out window's two endpoints comparing the SAME trade
        population, e.g. excluding a trade that matures mid-window rather
        than letting its NPV->0 transition register as a fake loss."""
        npv_dict = getattr(curve_result, curve)
        return sum(npv_dict[(path, date, tid)] for tid in self.trade_ids if tid not in exclude_trade_ids)
