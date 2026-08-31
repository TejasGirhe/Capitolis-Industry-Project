"""
Percentile fan charts (5/25/50/75/95) per risk factor, across all simulated
paths, over the fixed simulation grid -- shows the shape of the underlying
Monte Carlo distribution the exposure metrics are computed from.
"""
import math
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..factors.types import RateFactor, EquityFactor, FxFactor

PERCENTILES = (5, 25, 50, 75, 95)


def _rate_level_paths(precache, factor) -> np.ndarray:
    """Short-rate proxy: -ln(DF(0,t)) / t at each grid date -- an implied
    continuously-compounded zero rate from the model's own discount_factor(),
    not a separately-tracked short-rate series."""
    model = precache.rate_models[factor]
    cached = precache.rate_states[factor]   # (n_paths, n_dates, n_state)
    n_paths, n_dates, _ = cached.shape
    out = np.zeros((n_paths, n_dates))
    for d_idx, t in enumerate(precache.sim_times):
        if t <= 0:
            out[:, d_idx] = 0.0
            continue
        for p in range(n_paths):
            df = model.discount_factor(cached[p, d_idx, :], 0.0, t)
            out[p, d_idx] = -math.log(df) / t
    return out


def _spot_level_paths(precache, factor) -> np.ndarray:
    model = precache.spot_models[factor]
    cached = precache.spot_states[factor]
    n_paths, n_dates, _ = cached.shape
    rate_factor = next(iter(precache.rate_models)) if precache.rate_models else None
    rate_cached = precache.rate_states.get(rate_factor) if rate_factor else None
    out = np.zeros((n_paths, n_dates))
    for d_idx, t in enumerate(precache.sim_times):
        rate_state = rate_cached[:, d_idx, :] if rate_cached is not None else None
        for p in range(n_paths):
            rs = rate_state[p] if rate_state is not None else None
            out[p, d_idx] = model.level_at(cached[p, d_idx, :], t, rate_state=rs)
    return out


def plot_factor_fan(precache, factor, title: str, ylabel: str, out_path: str):
    if isinstance(factor, RateFactor):
        levels = _rate_level_paths(precache, factor)
    elif isinstance(factor, (EquityFactor, FxFactor)):
        levels = _spot_level_paths(precache, factor)
    else:
        raise TypeError(f"unrecognized factor type {type(factor)}")

    dates = precache.horizon_dates
    pct = {p: np.percentile(levels, p, axis=0) for p in PERCENTILES}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(dates, pct[5], pct[95], alpha=0.15, color="#3b6ea5", label="5-95%")
    ax.fill_between(dates, pct[25], pct[75], alpha=0.30, color="#3b6ea5", label="25-75%")
    ax.plot(dates, pct[50], color="#1f3a5f", linewidth=2, label="median")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Simulation date")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all_factor_fans(precache, factors, out_dir: str) -> List[str]:
    """One fan chart per factor (rate + a representative sample of equities +
    FX, since plotting all 37 equities individually isn't useful) -- caller
    picks which factors to include."""
    import os
    paths = []
    for factor in factors:
        if isinstance(factor, RateFactor):
            title, ylabel = f"USD short rate (implied) -- fan chart", "Rate"
            fname = f"factor_rate_{factor.currency}.png"
        elif isinstance(factor, EquityFactor):
            title, ylabel = f"Equity spot {factor.isin} -- fan chart", "Spot"
            fname = f"factor_equity_{factor.isin}.png"
        elif isinstance(factor, FxFactor):
            title, ylabel = f"FX {factor.base_ccy}{factor.quote_ccy} -- fan chart", "FX rate"
            fname = f"factor_fx_{factor.base_ccy}{factor.quote_ccy}.png"
        else:
            continue
        out_path = os.path.join(out_dir, fname)
        paths.append(plot_factor_fan(precache, factor, title, ylabel, out_path))
    return paths
