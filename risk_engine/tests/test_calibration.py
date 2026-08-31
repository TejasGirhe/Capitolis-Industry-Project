"""
Calibration sanity checks: recovered sigma(t)/theta reprice the input in the
simplest (flat-vol, single-strike) case before trusting calibration against a
real skewed surface. Also checks the "surface under-determined -> literature
prior" fallback path is actually taken/recorded when the surface has no smile.
"""
import os
import sys
from datetime import date

import pytest

RISK_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (RISK_ENGINE_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.calibration.market_surface import flat_vol_surface, VolSurface
from risk_engine.calibration.calibrate import calibrate
from risk_engine.models.registry import get_model


def test_flat_surface_sigma_matches_input_vol_exactly():
    flat_vol = 0.012
    vol_surface = flat_vol_surface("RATE_USD", flat_vol=flat_vol)
    calibrated = get_model("LGM1F").calibrate(_dummy_curve(), vol_surface)
    assert all(s == pytest.approx(flat_vol) for s in calibrated.sigma[0].sigmas)


def test_flat_surface_falls_back_to_sv_priors():
    """A degenerate single-strike surface has no smile to fit rho/eta from --
    calibration must fall back to the literature prior and record that fact,
    not silently fit noise from a single point."""
    vol_surface = flat_vol_surface("RATE_USD", flat_vol=0.012)
    calibrated = get_model("LGM1F_SV").calibrate(_dummy_curve(), vol_surface)
    assert calibrated.assumptions["eta_source"] == "literature prior (surface under-determined)"
    assert calibrated.assumptions["rho_source"] == "literature prior (surface under-determined)"
    assert calibrated.assumptions["theta_source"] == "ATM term vol level (identifiable)"


def test_smiled_surface_identifies_skew_and_curvature():
    """A real tenor x strike surface with a genuine skew/smile should be fit
    directly rather than falling back to the prior."""
    tenors, strikes = [1.0, 5.0], [0.9, 1.0, 1.1]
    vols = {
        (1.0, 0.9): 0.020, (1.0, 1.0): 0.015, (1.0, 1.1): 0.018,
        (5.0, 0.9): 0.022, (5.0, 1.0): 0.016, (5.0, 1.1): 0.019,
    }
    vol_surface = VolSurface(factor_key="RATE_USD", tenors=tenors, strikes=strikes, vols=vols)
    calibrated = get_model("LGM1F_SV").calibrate(_dummy_curve(), vol_surface)
    assert calibrated.assumptions["eta_source"] == "smile curvature"
    assert calibrated.assumptions["rho_source"] == "smile skew"


def test_calibrate_wrapper_matches_direct_model_call():
    vol_surface = flat_vol_surface("RATE_USD", flat_vol=0.01)
    curve = _dummy_curve()
    via_wrapper = calibrate("LGM2F", curve, vol_surface)
    direct = get_model("LGM2F").calibrate(curve, vol_surface)
    assert via_wrapper.mean_reversion == direct.mean_reversion


def _dummy_curve():
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(RISK_ENGINE_ROOT), "capitolis_pricers", "capitolis_pricers"))
    from capitolis_pricers.curves import zero_curve
    return zero_curve(date(2026, 1, 15), [0.5, 1, 2, 5, 10], [0.043, 0.042, 0.0405, 0.0395, 0.041])
