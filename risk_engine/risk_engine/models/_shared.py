"""
Shared calibration/simulation helpers used by every model family (LGM, GBM,
FXGBM). Kept in one place so the vol-bootstrap and smile-fit logic isn't
duplicated per factor kind -- rates, equity, and FX all calibrate sigma(t)
and the SV skew/curvature parameters the same way, against the same
VolSurface shape.
"""
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import List

import numpy as np


def mean_reversion_shift(a: float, u: float) -> float:
    """H(u) = (1 - exp(-a u)) / a, with the a->0 limit H(u) = u."""
    if abs(a) < 1e-8:
        return u
    return (1.0 - math.exp(-a * u)) / a


def bootstrap_sigma(atm_term_vol):
    """Piecewise-constant sigma(t) read directly off the surface's ATM term
    structure -- the curve/vol-fitting step shared by every factor kind.

    atm_term_vol: list of (tenor_years, vol) from VolSurface.atm_term_structure(),
    already sorted by tenor.
    """
    pillars, sigmas = [], []
    for tenor, vol in atm_term_vol:
        pillars.append(tenor)
        sigmas.append(max(vol, 1e-6))
    return pillars, sigmas


@dataclass
class PiecewiseSigma:
    pillars: List[float]
    sigmas: List[float]

    def at(self, t: float) -> float:
        if not self.pillars:
            return 0.0
        for p, s in zip(self.pillars, self.sigmas):
            if t <= p:
                return s
        return self.sigmas[-1]

    def zeta(self, t: float) -> float:
        """Accumulated variance integral_0^t sigma(s)^2 ds, piecewise-constant sigma."""
        if t <= 0 or not self.pillars:
            return 0.0
        prev_p, acc = 0.0, 0.0
        for p, s in zip(self.pillars, self.sigmas):
            seg_end = min(p, t)
            if seg_end > prev_p:
                acc += s * s * (seg_end - prev_p)
            prev_p = p
            if t <= p:
                return acc
        acc += self.sigmas[-1] ** 2 * (t - prev_p)
        return acc


def fit_theta(vol_surface):
    """Long-run variance level from the surface's longest-tenor ATM vol --
    always identifiable from a term structure, no smile needed."""
    atm = vol_surface.atm_term_structure()
    if not atm:
        return 0.04
    return atm[-1][1] ** 2


def fit_skew_smile(vol_surface):
    """Best-effort rho (skew) / eta (curvature) from the surface's smile at its
    longest tenor, if the surface actually has >=3 strikes (a real smile, not
    a degenerate single-strike flat surface). Returns (None, None) when there
    isn't enough strike resolution to identify them, so callers fall back to
    a literature prior explicitly rather than fitting noise.
    """
    if len(vol_surface.strikes) < 3:
        return None, None
    t = vol_surface.tenors[-1]
    ks = vol_surface.strikes
    vs = [vol_surface.vol(t, k) for k in ks]
    k_arr = np.array(ks) - 1.0
    v_arr = np.array(vs)
    if np.allclose(k_arr, 0.0):
        return None, None
    slope = float(np.polyfit(k_arr, v_arr, 1)[0])
    curvature = float(np.polyfit(k_arr, v_arr, 2)[0]) if len(ks) >= 3 else None
    rho = max(-0.99, min(0.99, -slope * 2.0))
    eta = None if curvature is None else max(0.05, min(1.5, abs(curvature) * 4.0))
    return rho, eta


def year_frac(ref_date, d):
    return (d - ref_date).days / 365.0


def shift_date(ref_date, years):
    return ref_date + timedelta(days=round(years * 365.0))


def simulate_cir_variance_step(v_prev, kappa, eta, dt, z):
    """One Euler step of a mean-1 CIR variance multiplier:
    dv = kappa*(1-v)dt + eta*sqrt(v)*dW. Shared by every -SV model's
    stepping loop (LGM*_SV, GBM_SV, FXGBM_SV) so the (documented) O(dt)
    discretization behavior is identical across factor kinds."""
    dv = kappa * (1.0 - v_prev) * dt + eta * np.sqrt(np.maximum(v_prev, 0.0) * dt) * z
    return np.maximum(v_prev + dv, 1e-10)
