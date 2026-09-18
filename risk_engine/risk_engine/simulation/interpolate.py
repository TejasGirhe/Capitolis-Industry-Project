"""
Brownian-bridge interpolation: given a cached path simulated only on a fixed
simulation grid, get the state at an arbitrary date between two grid dates
without a fresh Monte Carlo draw.

For a driftless Brownian motion x(t) with accumulated variance zeta(t)
(exactly LGM's state, and GBM/FXGBM's log_growth martingale term -- see
models/lgm.py and models/gbm.py module docstrings), the bridge between two
observed points x(t0), x(t1) has conditional mean AND variance

    E[x(d) | x(t0), x(t1)] = x(t0) + w * (x(t1) - x(t0)),   w = (zeta(d)-zeta(t0)) / (zeta(t1)-zeta(t0))
    Var[x(d) | x(t0), x(t1)] = (zeta(d)-zeta(t0)) * (zeta(t1)-zeta(d)) / (zeta(t1)-zeta(t0))

i.e. linear interpolation weighted by ACCUMULATED VARIANCE, not raw calendar
time (raw-time interpolation would be wrong whenever sigma(t) varies -- e.g.
across a vol-surface term structure with different vol in different
periods).

interpolate_state() adds the exact conditional-variance bridge noise on
top of the conditional mean by DEFAULT (add_bridge_noise=True) -- see
bridge_conditional_variance / draw_bridge_noise below for how the noise
term is constructed as a REPRODUCIBLE per-(path, t) draw (needed because
this function is called repeatedly for the same (path, factor, t) across
independent multiprocess pricing workers, with no RNG state threaded
through that call chain). Passing add_bridge_noise=False instead returns
only the conditional MEAN -- the OLD default, kept available for
regression comparison (see examples/bridge_vs_full_grid.py) but no longer
what production pricing uses. The bridge's conditional variance is
greatest at the midpoint of a gap and zero at the grid dates themselves,
so with the mean-only mode, off-grid exposure metrics silently understated
variance in proportion to how sparse the simulation grid is relative to
the reporting/regression dates actually queried (28 grid dates vs. ~94
regression dates in this book) -- switched to the full bridge by default
so exposure numbers reflect genuine within-interval uncertainty rather
than a smoothed/deterministic interpolation.
"""
import hashlib
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


def bridge_conditional_variance(zeta_fn: Callable[[float], float], t0: float, t1: float, d: float) -> float:
    """Var[x(d) | x(t0), x(t1)] for a driftless Brownian motion with
    accumulated variance zeta(t) -- the bridge noise interpolate_state's
    mean-only mode omits. Zero at t0/t1 (the bridge is pinned at both
    observed endpoints), maximized at the zeta-midpoint between them."""
    if t1 <= t0:
        return 0.0
    z0, z1, zd = zeta_fn(t0), zeta_fn(t1), zeta_fn(d)
    if z1 <= z0:
        return 0.0
    zd = max(z0, min(z1, zd))
    return (zd - z0) * (z1 - zd) / (z1 - z0)


def draw_bridge_noise(path_ids: np.ndarray, t: float, seed_salt: str, n_drivers: int = 1,
                       corr_chol: np.ndarray = None) -> np.ndarray:
    """Reproducible standard-normal bridge noise, shape (len(path_ids), n_drivers).

    interpolate_state() is called repeatedly for the SAME (path, factor, t)
    -- e.g. once for npv0 at a regression date and again for npv10 at that
    date +10bd, and independently by every trade/date iteration in
    price_curves' multiprocess path-chunk workers. There is no RNG state
    threaded into that call chain (state_at() is a pure function of the
    cached path), so bridge noise cannot be a fresh rng.standard_normal()
    draw without breaking reproducibility across those repeated calls.

    Instead each path's noise is derived deterministically from a hash of
    (path_id, t, seed_salt): seed a per-path np.random.default_rng from that
    hash, draw n_drivers standard normals. Reproducible (same inputs ->
    same output, safe to call from any worker process) and independent
    across paths and across distinct t values.

    corr_chol: optional (n_drivers, n_drivers) Cholesky factor to correlate
    the independent draw via z @ corr_chol.T (rho = corr_chol @ corr_chol.T)
    -- pass the FULL cross-factor correlation Cholesky factor with
    n_drivers = total driver count to get ONE jointly-correlated draw across
    every factor at once (see build_market_states_at, which slices the
    result per factor afterward); this is the only way to get bridge noise
    correlated ACROSS factors correctly -- correlating a per-factor
    sub-block separately does not reproduce the true cross-factor
    correlation unless that factor happens to be first in driver order
    (confirmed directly: slicing the full Cholesky factor L at a nonzero
    offset reconstructs a materially wrong diagonal block, e.g.
    [[0.907,0.547],[0.547,0.827]] instead of the true [[1,0.6],[0.6,1]] in a
    4x4 test case). seed_salt should be one FIXED value per logical joint
    draw (e.g. "joint_bridge"), not per-factor, since this is one shared
    draw sliced afterward, not independent per-factor draws.
    """
    n_paths = len(path_ids)
    z = np.empty((n_paths, n_drivers))
    for i, pid in enumerate(path_ids):
        digest = hashlib.sha256(f"{seed_salt}|{int(pid)}|{t!r}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        z[i, :] = rng.standard_normal(n_drivers)
    if corr_chol is not None:
        z = z @ corr_chol.T
    return z


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
                       zeta_fn: Callable[[float], float], add_bridge_noise: bool = True,
                       seed_salt: str = "", external_z: np.ndarray = None) -> np.ndarray:
    """cached_path: (n_paths, n_sim_dates, n_state_vars) -- one factor's full
    simulated path on the fixed simulation grid. sim_times: year-fractions of
    each grid date, same length as cached_path's date axis. t: target
    year-fraction (need not be on the grid). zeta_fn: this factor's own
    accumulated-variance function, e.g. a PiecewiseSigma.zeta or an
    equivalent per-model callable.

    Returns (n_paths, n_state_vars) -- the interpolated state at t, exact
    (no interpolation) when t lands exactly on a grid date.

    add_bridge_noise: True (default) adds the exact conditional-variance
    bridge noise on top of the conditional mean (see module docstring and
    bridge_conditional_variance) -- only the FIRST state column (the driven
    Brownian dimension, e.g. LGM's x or GBM's log_growth) gets noise added;
    auxiliary columns (zeta bookkeeping, LGM2F's second state var handled by
    its own call to this function) are left at their mean. Pass False to
    get the OLD mean-only behavior (kept for regression comparison, see
    examples/bridge_vs_full_grid.py).

    external_z: (n_paths,) standard-normal noise, ALREADY correlated across
    factors if that matters to the caller (see
    scenario_market.build_market_states_at, which draws one joint vector via
    draw_bridge_noise and slices it per factor before calling this). None
    (default, when add_bridge_noise=True) falls back to an independent
    draw_bridge_noise call keyed on seed_salt -- correct for a standalone
    single-factor use, but callers needing cross-factor correlation must
    pass their own pre-correlated external_z instead.
    """
    i0, i1, t0, t1 = bracket_grid_dates(sim_times, t)
    if i0 == i1:
        return cached_path[:, i0, :]
    w = bridge_weight(zeta_fn, t0, t1, t)
    mean = (1.0 - w) * cached_path[:, i0, :] + w * cached_path[:, i1, :]
    if not add_bridge_noise:
        return mean
    var = bridge_conditional_variance(zeta_fn, t0, t1, t)
    if var <= 0.0:
        return mean
    n_paths = cached_path.shape[0]
    z = external_z if external_z is not None else draw_bridge_noise(np.arange(n_paths), t, seed_salt, n_drivers=1)[:, 0]
    noise = np.sqrt(var) * z
    mean = mean.copy()
    mean[:, 0] += noise
    return mean
