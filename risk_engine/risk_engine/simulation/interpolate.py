"""
Conditional-mean Brownian-bridge interpolation: given a cached path simulated
only on a fixed simulation grid, get the state at an arbitrary date between
two grid dates without a fresh Monte Carlo draw.

For a driftless Brownian motion x(t) with accumulated variance zeta(t)
(exactly LGM's state, and GBM/FXGBM's log_growth martingale term -- see
models/lgm.py and models/gbm.py module docstrings), the bridge between two
observed points x(t0), x(t1) has conditional mean

    E[x(d) | x(t0), x(t1)] = x(t0) + w * (x(t1) - x(t0)),   w = (zeta(d)-zeta(t0)) / (zeta(t1)-zeta(t0))

i.e. linear interpolation weighted by ACCUMULATED VARIANCE, not raw calendar
time (raw-time interpolation would be wrong whenever sigma(t) varies -- e.g.
across a vol-surface term structure with different vol in different
periods). This module deliberately returns only the conditional MEAN (no
bridge noise added) -- confirmed design: deterministic given the cached
path, reproducible, and standard for a reporting grid dense enough that
off-grid points are never far from a real simulated node. The bridge's
conditional VARIANCE (the noise this omits) is greatest at the midpoint of a
gap and zero at the grid dates themselves, so grid density controls how much
this approximation understates variance between nodes.
"""
from bisect import bisect_left
from typing import Callable, List

import numpy as np


def bridge_weight(zeta_fn: Callable[[float], float], t0: float, t1: float, d: float) -> float:
    """w in [0,1] such that E[x(d)] = (1-w)*x(t0) + w*x(t1), weighted by
    accumulated variance zeta rather than raw time."""
    if t1 <= t0:
        return 0.0
    z0, z1, zd = zeta_fn(t0), zeta_fn(t1), zeta_fn(d)
    if z1 <= z0:
        # zero-vol segment (e.g. before any pillar) -- fall back to linear
        # time-weighting so a degenerate zeta doesn't produce a NaN weight.
        return (d - t0) / (t1 - t0)
    return max(0.0, min(1.0, (zd - z0) / (z1 - z0)))


def bracket_grid_dates(sim_times: List[float], t: float):
    """(i0, i1, t0, t1) -- indices and times of the two simulation-grid dates
    bracketing t, clamped at the ends (t before the first date or after the
    last uses that endpoint for both, i.e. no extrapolation)."""
    n = len(sim_times)
    if t <= sim_times[0]:
        return 0, 0, sim_times[0], sim_times[0]
    if t >= sim_times[-1]:
        return n - 1, n - 1, sim_times[-1], sim_times[-1]
    i1 = bisect_left(sim_times, t)
    if sim_times[i1] == t:
        return i1, i1, t, t
    i0 = i1 - 1
    return i0, i1, sim_times[i0], sim_times[i1]


def interpolate_state(cached_path: np.ndarray, sim_times: List[float], t: float,
                       zeta_fn: Callable[[float], float]) -> np.ndarray:
    """cached_path: (n_paths, n_sim_dates, n_state_vars) -- one factor's full
    simulated path on the fixed simulation grid. sim_times: year-fractions of
    each grid date, same length as cached_path's date axis. t: target
    year-fraction (need not be on the grid). zeta_fn: this factor's own
    accumulated-variance function, e.g. a PiecewiseSigma.zeta or an
    equivalent per-model callable.

    Returns (n_paths, n_state_vars) -- the interpolated (conditional-mean)
    state at t. Exact (no interpolation) when t lands exactly on a grid date.
    """
    i0, i1, t0, t1 = bracket_grid_dates(sim_times, t)
    if i0 == i1:
        return cached_path[:, i0, :]
    w = bridge_weight(zeta_fn, t0, t1, t)
    return (1.0 - w) * cached_path[:, i0, :] + w * cached_path[:, i1, :]
