"""
Forward-matching test: for each LGM variant, calibrate to a sample curve and
assert that E[DF(t,T)] -- the model's own closed-form DF(t,T) formula,
averaged over simulated x(t) across paths -- reproduces the deterministic
curve ratio DF(0,T)/DF(0,t), within MC noise. This is the LGM martingale
property under its own numeraire (the actual content of "forward matching"):
picking a future pricing time t, the *expected* discount factor to any later
T must equal today's forward discount ratio implied by the input curve. A
failure here means the calibration or DF formula has a real bug, not that the
MC sampling is noisy.

(A DF(0,T) query with a zero state is deterministic and trivially matches the
curve by construction -- see test_df_t_t_is_one_identically for the other
trivial identity, DF(t,t)==1. Neither of those exercises the stochastic part
of the model, which is why this test instead evaluates DF(t,T) for T>t on the
simulated x(t).)
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pytest

RISK_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICERS_ROOT = os.path.join(os.path.dirname(RISK_ENGINE_ROOT), "capitolis_pricers", "capitolis_pricers")
for p in (RISK_ENGINE_ROOT, PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from capitolis_pricers.curves import zero_curve

from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.models.registry import get_model, MODEL_REGISTRY


REF_DATE = date(2026, 1, 15)
CURVE = zero_curve(REF_DATE, [0.5, 1, 2, 5, 10], [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])
# (t, T) pairs: simulate to t, price a bond maturing at T > t
TEST_PAIRS_YEARS = [(0.25, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0)]
N_PATHS = 50_000
SEED = 7


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_expected_df_matches_forward_curve_ratio(model_name):
    """E[DF(t,T)], averaged over simulated x(t), matches DF(0,T)/DF(0,t)."""
    vol_surface = flat_vol_surface("RATE_USD", flat_vol=0.010)
    model = get_model(model_name)
    calibrated = model.calibrate(CURVE, vol_surface)

    sim_times = sorted({t for t, _ in TEST_PAIRS_YEARS})
    horizon_dates = [REF_DATE + timedelta(days=round(t * 365)) for t in sim_times]
    rng = np.random.default_rng(SEED)
    paths = calibrated.simulate_paths(N_PATHS, horizon_dates, rng)  # (n_paths, n_dates, n_factors)
    time_index = {t: i for i, t in enumerate(sim_times)}

    for t, T in TEST_PAIRS_YEARS:
        state_at_t = paths[:, time_index[t], :]
        simulated_df_tT = np.array([
            calibrated.discount_factor(state_at_t[p], t, T) for p in range(N_PATHS)
        ])
        df0_t = CURVE.discount(REF_DATE + timedelta(days=round(t * 365)))
        df0_T = CURVE.discount(REF_DATE + timedelta(days=round(T * 365)))
        expected_fwd_ratio = df0_T / df0_t

        mc_mean = simulated_df_tT.mean()
        mc_stderr = simulated_df_tT.std() / np.sqrt(N_PATHS)
        # SV variants use an Euler scheme for the CIR variance process, which
        # has O(dt) discretization bias (unlike the LGM state itself, which is
        # driftless Brownian and exact at any step size) -- so their tolerance
        # needs a small absolute floor on top of the MC stderr multiple, calibrated
        # to the discount factor's own scale rather than a fixed constant.
        tol = max(6 * mc_stderr, 2e-4 * expected_fwd_ratio)
        assert abs(mc_mean - expected_fwd_ratio) < tol, (
            f"{model_name} @ (t={t}, T={T}): MC mean DF(t,T)={mc_mean:.6f} vs "
            f"DF(0,T)/DF(0,t)={expected_fwd_ratio:.6f} (stderr={mc_stderr:.2e}, tol={tol:.2e})"
        )


def test_df_t_t_is_one_identically():
    """Sanity check: the DF formula must give DF(t,t) == 1 for any state,
    since H(0) == 0 regardless of mean reversion."""
    vol_surface = flat_vol_surface("RATE_USD", flat_vol=0.010)
    calibrated = get_model("LGM2F_SV").calibrate(CURVE, vol_surface)
    for state in (np.zeros(2), np.array([0.01, -0.02]), np.array([-0.5, 0.3])):
        assert calibrated.discount_factor(state, 1.5, 1.5) == pytest.approx(1.0, abs=1e-9)


def test_df_0_T_from_zero_state_matches_curve_exactly():
    """A DF(0,T) query with the zero state is deterministic (no stochastic
    term contributes at t=0, zeta(0)=0) -- must reproduce the input curve
    exactly, not just within MC tolerance."""
    vol_surface = flat_vol_surface("RATE_USD", flat_vol=0.010)
    for model_name in MODEL_REGISTRY:
        calibrated = get_model(model_name).calibrate(CURVE, vol_surface)
        for T in [0.5, 1.0, 5.0, 10.0]:
            df = calibrated.discount_factor(np.zeros(calibrated.n_factors), 0.0, T)
            expected = CURVE.discount(REF_DATE + timedelta(days=round(T * 365)))
            assert df == pytest.approx(expected, rel=1e-9), model_name
