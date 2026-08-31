"""
The one simulation entry point: pick a model by name, calibrate it to market,
simulate paths. This is the "pass a model name to a function" contract.

    from risk_engine import simulate
    result = simulate("LGM2F_SV", rate_factor, curve, vol_surface,
                       n_paths=10_000, horizon_dates=dates)
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .calibration.calibrate import calibrate


@dataclass
class SimResult:
    factor: object
    model_name: str
    paths: object              # (n_paths, len(horizon_dates), n_factors) state array
    calibrated_model: object   # holds discount_factor() for repricing on each path


def simulate(model_name: str, factor, curve, vol_surface, n_paths: int,
             horizon_dates: List, rng=None, model_kwargs=None, **calibrate_kwargs) -> SimResult:
    """Calibrate `model_name` to `curve`/`vol_surface` and simulate `n_paths`
    trajectories of `factor` out to each date in `horizon_dates`.

    factor: the RiskFactor this simulation drives (kept on the result for
        bookkeeping/correlation-matrix indexing by callers).
    curve: capitolis_pricers.curves.Curve for the factor's currency.
    vol_surface: risk_engine.calibration.market_surface.VolSurface for this factor.
    """
    calibrated = calibrate(model_name, curve, vol_surface, model_kwargs=model_kwargs, **calibrate_kwargs)
    rng = rng or np.random.default_rng()
    paths = calibrated.simulate_paths(n_paths, horizon_dates, rng)
    return SimResult(factor=factor, model_name=model_name, paths=paths, calibrated_model=calibrated)
