"""
MarginModel: pluggable collateral/margin layer on top of netted exposure.
ZeroMargin (the default -- no collateral posted/held) makes
collateralized_exposure == netted_npv, matching Slide 9's stated assumption
("assume no initial margin"). Future variants (VariationMarginModel,
ThresholdMarginModel, ...) implement the same interface without touching
NettingSet/Counterparty/exposure.py.

Note this is a DIFFERENT thing from the rolling 10-day close-out window in
exposure.py, which already encodes Slide 9's own VM rule (VM = prior-day NPV,
discontinues at default -- that's baked into the exposure() differencing
itself, not into this MarginModel). This plug point is for genuine
collateral actually posted/held on top of that, e.g. independent amount,
threshold/MTA-gated variation margin, initial margin -- none of which this
book's assumptions currently require (Slide 9: "assume no initial margin").
"""
from abc import ABC, abstractmethod


class MarginModel(ABC):
    @abstractmethod
    def posted_collateral(self, netting_set, curve_result, curve: str, path: int, date) -> float:
        """Collateral held against netting_set's exposure at this node,
        i.e. the amount subtracted from netted_npv to get collateralized
        exposure. Positive = collateral reduces exposure.

        curve_result: risk_engine.pricing.CurveResult. curve: 'npv0' or
        'npv10' -- which of the two curves this collateral check applies to."""
        raise NotImplementedError


class ZeroMargin(MarginModel):
    """No collateral/margin posted -- Slide 9's stated baseline assumption."""

    def posted_collateral(self, netting_set, curve_result, curve: str, path: int, date) -> float:
        return 0.0
