"""
NettingSet: a group of trades netted together for exposure purposes.

netting_set.id is now a DUMMY netting-agreement id, independent of (though
currently still 1:1 derived from) counterparty id -- see
build_netting_hierarchy's DUMMY_NETTING_AGREEMENT_SUFFIX. This project has
no real ISDA/CSA agreement data (no per-counterparty agreement count,
product-eligibility, or legal-netting-opinion feed), so a real multi-
agreement-per-counterparty book cannot be represented yet -- but the id is
no longer a bare alias for counterparty, so a real agreement feed can be
wired in later by changing build_netting_hierarchy's grouping key alone,
without touching NettingSet, Counterparty, or any exposure/pricing code
that reads netting_set.id. Confirmed as a stated simplification (see
risk_engine/examples/regulatory_readiness_report.html Sec.05), not
something this change claims to have solved.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class NettingSet:
    id: str            # DUMMY netting-agreement id (see module docstring), not a counterparty id
    counterparty_id: str
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
