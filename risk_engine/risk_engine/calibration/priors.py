"""
Literature-typical priors for stochastic-vol parameters that the book's vol
surface cannot fully identify on its own (no swaption/cap grid is quoted --
MARKET_DATA.md Sec.5 gives, at best, a per-tenor term structure; the extended
surface in market_surface.py adds a strike axis but the resulting smile is
still thin evidence for a full CIR/Heston fit).

Calibration policy (confirmed with the user): fit whatever IS identifiable
from the surface -- ATM term vol -> sigma(t); skew slope -> rho; smile
curvature -> eta -- and fall back to these priors for anything left
under-determined, rather than silently defaulting or refusing to run.
Every fallback is recorded on the CalibratedModel so it shows up in the
technical report rather than being a hidden assumption.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SVPriors:
    kappa: float   # mean-reversion speed of variance
    theta: float   # long-run variance level (annualized vol^2)
    eta: float     # vol-of-vol
    rho: float     # correlation(rate factor, variance)


# Rates stochastic-vol priors (CIR variance on top of LGM), Heston-style rate
# literature typical ranges: kappa ~ 1-3, eta ~ 0.3-0.5, rho ~ -0.3 (rates/vol
# tend to move opposite: rates up -> vol down, similar to equity leverage effect
# but weaker and less consistently signed than equity).
RATE_SV_PRIOR = SVPriors(kappa=1.5, theta=None, eta=0.40, rho=-0.30)  # theta set from ATM term vol at calibration time

# Equity stochastic-vol priors (Heston), typical single-name/index calibration
# ranges: kappa ~ 1-2, eta ~ 0.5-0.8, rho ~ -0.6 to -0.8 (leverage effect).
EQUITY_SV_PRIOR = SVPriors(kappa=1.5, theta=None, eta=0.60, rho=-0.70)

FX_SV_PRIOR = SVPriors(kappa=1.0, theta=None, eta=0.35, rho=0.0)


def rate_sv_prior(theta: float) -> SVPriors:
    """RATE_SV_PRIOR with theta (long-run variance) set from the surface's ATM
    term vol -- the one SV parameter always identifiable from a term structure."""
    return SVPriors(kappa=RATE_SV_PRIOR.kappa, theta=theta, eta=RATE_SV_PRIOR.eta, rho=RATE_SV_PRIOR.rho)
