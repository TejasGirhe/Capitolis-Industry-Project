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
    """Second, independent call site with the SAME single-rate-factor
    assumption scenario_market.build_market_states_at used to have (fixed
    when a real JPY rate factor was added alongside USD -- see that
    module's drift_rate_factor_by_spot/foreign_rate_factor_by_spot). This
    one was missed in that pass and broke with a real ValueError the first
    time a factor fan chart was plotted for a book with a real foreign FX
    rate factor: next(iter(precache.rate_models)) picks an ARBITRARY rate
    factor (dict order), which is ambiguous once USD and JPY both exist,
    and never looked up a foreign leg at all. Fixed the same way: use the
    explicit drift/foreign mappings JointSimResult now carries, with the
    old "only one rate factor" behavior kept as a fallback for a precache
    that predates those mappings (e.g. a cached/pickled older result)."""
    model = precache.spot_models[factor]
    cached = precache.spot_states[factor]
    n_paths, n_dates, _ = cached.shape

    drift_map = getattr(precache, "drift_rate_factor_by_spot", None) or {}
    foreign_map = getattr(precache, "foreign_rate_factor_by_spot", None) or {}
    rate_factor = drift_map.get(factor)
    if rate_factor is None:
        rate_factor = next(iter(precache.rate_models)) if precache.rate_models else None
    foreign_factor = foreign_map.get(factor)

    rate_cached = precache.rate_states.get(rate_factor) if rate_factor else None
    foreign_cached = precache.rate_states.get(foreign_factor) if foreign_factor else None

    out = np.zeros((n_paths, n_dates))
    for d_idx, t in enumerate(precache.sim_times):
        rate_state = rate_cached[:, d_idx, :] if rate_cached is not None else None
        foreign_state = foreign_cached[:, d_idx, :] if foreign_cached is not None else None
        for p in range(n_paths):
            rs = rate_state[p] if rate_state is not None else None
            kwargs = {}
            if foreign_state is not None:
                kwargs["foreign_rate_state"] = foreign_state[p]
            out[p, d_idx] = model.level_at(cached[p, d_idx, :], t, rate_state=rs, **kwargs)
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
