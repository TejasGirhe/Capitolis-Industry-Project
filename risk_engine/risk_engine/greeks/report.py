"""
GreeksReport: bundles SA-CCR delta (Basel III Annex 4, a regulatory-capital
INPUT) and bump-and-reprice vega (this session's own risk-desk sensitivity,
NOT a Basel III concept) into one object, kept visually and structurally
separate so the two are never confused for one another.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GreeksReport:
    ref_date: object
    sa_ccr_delta: Dict[str, Dict[str, float]]              # {counterparty: {asset_class: delta}}
    vega_runs: List[Dict] = field(default_factory=list)     # list of bump_vol_and_reprice(...) return dicts

    def summary(self) -> str:
        lines = []
        lines.append(f"=== SA-CCR Delta (Basel III Annex 4, regulatory) -- as of {self.ref_date} ===")
        lines.append("(feeds EAD = alpha * (RC + PFE_addon), Slide 15 -- per-counterparty, per-asset-class "
                      "effective notional; full hedging-set AddOn aggregation is out of scope, see sa_ccr.py)")
        for cpty in sorted(self.sa_ccr_delta):
            lines.append(f"  {cpty}:")
            for asset_class, delta in sorted(self.sa_ccr_delta[cpty].items()):
                lines.append(f"    {asset_class:20s} {delta:>18,.2f}")

        lines.append("")
        lines.append("=== Vega (bump-and-reprice sensitivity -- NOT a Basel III / SA-CCR input) ===")
        lines.append("(SA-CCR has no vega concept -- its PFE multiplier uses a fixed supervisory vol "
                      "factor per asset class, not a shocked recompute; this is a risk-desk metric only)")
        for run in self.vega_runs:
            lines.append(f"  Factor group: {run['factor_group']}  (bump={run['bump_size']:+.4f}, "
                         f"{run['elapsed_seconds']:.1f}s)")
            for cpty in sorted(run["vega"]):
                v = run["vega"][cpty]
                lines.append(f"    {cpty:12s} dEE_max={v['EE_max']:>14,.2f}  dMPE_95={v['MPE_95']:>14,.2f}  "
                             f"dMPE_99={v['MPE_99']:>14,.2f}  dEEPE={v['EEPE']:>14,.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ref_date": self.ref_date.isoformat() if hasattr(self.ref_date, "isoformat") else str(self.ref_date),
            "sa_ccr_delta": self.sa_ccr_delta,
            "vega_runs": self.vega_runs,
        }
