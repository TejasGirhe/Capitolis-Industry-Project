"""
CVA, DVA, FVA -- computed from an ExposureProfile's already-simulated
EE(t)/NEE(t) curves (risk_engine.exposure), discounted by the already-
calibrated USD curve, weighted by default probabilities from a
capitolis_pricers.credit.CreditCurve (see risk_engine.market_data.credit for
how those curves are sourced -- a rating-tier proxy, not a real
counterparty-specific CDS curve, since none exists free).

Standard unilateral formulas, summed over the ExposureProfile's own anchor
dates (no new date grid -- reuses exactly what Slide 8/9's EE/PFE engine
already computed):

    CVA = (1 - R_cpty) * sum_t DF(0,t) * EE(t)  * PD_cpty(t_prev, t)
    DVA = (1 - R_own)  * sum_t DF(0,t) * |NEE(t)| * PD_own(t_prev, t)
    FVA = sum_t DF(0,t) * EE(t) * funding_spread * dt(t_prev, t)

CVA reduces the value of the trade to us (counterparty might not pay);
DVA increases it (symmetric benefit: we might not pay either); FVA is a
funding cost of carrying uncollateralized exposure. Sign convention:
net_xva = cva - dva + fva -- the total valuation adjustment SUBTRACTED
from the book's risk-free value (so a positive net_xva means net cost).

t_prev, for the first anchor date, is ref_date itself (PD/dt measured from
today, not from a nonexistent prior anchor).
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class XVAResult:
    owner_id: str
    cva: float
    dva: float
    fva: float

    @property
    def net_xva(self) -> float:
        return self.cva - self.dva + self.fva


def _anchor_pairs(dates: List, ref_date):
    """[(t_prev, t, dt_years), ...] -- t_prev is ref_date for the first
    anchor, the prior anchor date for every subsequent one."""
    prev = ref_date
    out = []
    for t in dates:
        dt_years = (t - prev).days / 365.0
        out.append((prev, t, dt_years))
        prev = t
    return out


def compute_cva(profile, credit_curve, discount_curve, ref_date) -> float:
    """CVA = (1-R) * sum_t DF(0,t) * EE(t) * PD(t_prev, t)."""
    recovery = credit_curve.recovery
    cva = 0.0
    for (t_prev, t, _dt), ee_t in zip(_anchor_pairs(profile.dates, ref_date), profile.ee):
        if ee_t <= 0.0:
            continue
        pd = credit_curve.default_prob(t_prev, t)
        df = discount_curve.discount(t)
        cva += (1.0 - recovery) * df * ee_t * pd
    return cva


def compute_dva(profile, own_credit_curve, discount_curve, ref_date) -> float:
    """DVA = (1-R_own) * sum_t DF(0,t) * |NEE(t)| * PD_own(t_prev, t)."""
    recovery = own_credit_curve.recovery
    dva = 0.0
    for (t_prev, t, _dt), nee_t in zip(_anchor_pairs(profile.dates, ref_date), profile.nee):
        if nee_t >= 0.0:
            continue
        pd = own_credit_curve.default_prob(t_prev, t)
        df = discount_curve.discount(t)
        dva += (1.0 - recovery) * df * abs(nee_t) * pd
    return dva


def compute_fva(profile, funding_spread: float, discount_curve, ref_date) -> float:
    """FVA = sum_t DF(0,t) * EE(t) * funding_spread * dt(t_prev, t)."""
    fva = 0.0
    for (t_prev, t, dt_years), ee_t in zip(_anchor_pairs(profile.dates, ref_date), profile.ee):
        if ee_t <= 0.0 or dt_years <= 0.0:
            continue
        df = discount_curve.discount(t)
        fva += df * ee_t * funding_spread * dt_years
    return fva


def compute_xva_report(profile, counterparty_credit_curve, own_credit_curve, funding_spread: float,
                        discount_curve, ref_date) -> XVAResult:
    cva = compute_cva(profile, counterparty_credit_curve, discount_curve, ref_date)
    dva = compute_dva(profile, own_credit_curve, discount_curve, ref_date)
    fva = compute_fva(profile, funding_spread, discount_curve, ref_date)
    return XVAResult(owner_id=profile.owner_id, cva=cva, dva=dva, fva=fva)


def compute_all_xva(profiles: Dict[str, object], credit_curves_by_cpty: Dict[str, object],
                     own_credit_curve, funding_spread: float, discount_curve, ref_date) -> Dict[str, XVAResult]:
    """{counterparty_id: XVAResult} -- mirrors exposure.compute_all_profiles'
    shape. `profiles`: {counterparty_id: ExposureProfile} (e.g.
    exposure.compute_all_profiles' output, minus BOOK_TOTAL which has no
    single counterparty credit curve to apply -- pass only the per-
    counterparty entries, not the BOOK_TOTAL key)."""
    out = {}
    for cpty, profile in profiles.items():
        if cpty not in credit_curves_by_cpty:
            continue
        out[cpty] = compute_xva_report(profile, credit_curves_by_cpty[cpty], own_credit_curve,
                                        funding_spread, discount_curve, ref_date)
    return out
