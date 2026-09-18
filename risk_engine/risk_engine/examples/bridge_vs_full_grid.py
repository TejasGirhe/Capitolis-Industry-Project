"""
Section 6/7 of the simulation-framework deep-dive: controlled comparison of
Method A (full daily-grid simulation) vs. Method B (coarse grid + Brownian-
bridge interpolation), for ONE representative rate factor (USD) and ONE
representative equity/FX factor -- per the confirmed scope (a minimal,
non-invasive comparison, not a rebuild of the production JointSimulator to
run daily by default).

IMPORTANT CAVEAT, stated explicitly rather than glossed over: "same seed"
does NOT mean "identical sample paths" here. JointSimulator.simulate()
draws rng.standard_normal((n_paths, n_steps, n_drivers)) where n_steps
depends on len(horizon_dates) -- a daily grid to a ~2.5-year horizon has
~900 steps, the production coarse grid has 28. Even with the identical
seed, these two calls consume the SAME underlying PCG64 stream but reshape
it into DIFFERENT arrays, so path 0 in the daily-grid run and path 0 in
the coarse-grid run are NOT the same Brownian path realization sampled at
different resolutions -- they are two DIFFERENT (but equally valid, same-
distribution) sets of paths. This is a fundamental consequence of how
pseudo-random streams are consumed, not a bug in either code path.

Consequently this comparison is a STATISTICAL one (do the two methods'
path ensembles have statistically indistinguishable moments/quantiles at
the SAME calendar dates), not a per-path deterministic-equality check.
This is stated explicitly per the requirement to flag exactly what
approximation/limitation applies.

Method A (full grid): simulate every date in DAILY_DATES directly --
the model's own state IS available at every date queried, no
interpolation needed at all.

Method B (bridge): simulate ONLY the production coarse grid, then use
state_at() (Brownian-bridge, conditional MEAN only -- current production
default) to obtain the state at each of DAILY_DATES's dates, matching
Method A's date set for a fair comparison at IDENTICAL calendar points.

    python risk_engine/examples/bridge_vs_full_grid.py
"""
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import date, timedelta

import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.models._shared import year_frac

REF_DATE = date(2026, 8, 31)
N_PATHS = 5000
SEED = 777
HORIZON_YEARS = 2.5   # matches roughly this book's own horizon scale
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "bridge_vs_full_grid_results.json")


def _daily_dates(ref_date, horizon_years):
    n_days = int(horizon_years * 365)
    return [ref_date + timedelta(days=d) for d in range(1, n_days + 1)]


def _coarse_grid(ref_date, horizon_years):
    """The SAME coarse-grid density production actually uses (weekly/month3,
    monthly/year1, quarterly beyond) -- reuses simulation.grid.build_simulation_grid
    directly rather than re-deriving the density rule, so Method B is a
    faithful stand-in for the real production grid, not an invented one."""
    from risk_engine.simulation.grid import build_simulation_grid
    max_maturity = ref_date + timedelta(days=int(horizon_years * 365))
    grid = build_simulation_grid(ref_date, max_maturity)
    return [d for d in grid.dates if d > ref_date]


def _comparison_dates(ref_date, horizon_years, n_points=40):
    """A representative subset of calendar dates (evenly spaced across the
    horizon) at which BOTH methods will be evaluated -- these need not be
    on either method's own native grid; Method A gets them for free (daily
    grid contains every date), Method B reaches them via the bridge."""
    days = np.linspace(30, int(horizon_years * 365) - 5, n_points).astype(int)
    return [ref_date + timedelta(days=int(d)) for d in days]


def method_a_full_grid(sim, market_state, n_paths, ref_date, horizon_years, rng_seed):
    """Simulate the USD rate factor and one equity factor DIRECTLY on a
    daily grid -- every comparison date is a real simulated date, zero
    interpolation."""
    daily_dates = _daily_dates(ref_date, horizon_years)
    rng = np.random.default_rng(rng_seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    precache = sim.simulate(market_state, n_paths=n_paths, horizon_dates=daily_dates, rng=rng, ref_date=ref_date)
    sim_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return precache, daily_dates, sim_time, peak / 1e6


def method_b_bridge(sim, market_state, n_paths, ref_date, horizon_years, rng_seed):
    """Simulate ONLY the production coarse grid, rely on state_at()'s
    Brownian-bridge (conditional mean, current production default --
    add_bridge_noise=False) for every other date."""
    coarse_dates = _coarse_grid(ref_date, horizon_years)
    rng = np.random.default_rng(rng_seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    precache = sim.simulate(market_state, n_paths=n_paths, horizon_dates=coarse_dates, rng=rng, ref_date=ref_date)
    sim_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return precache, coarse_dates, sim_time, peak / 1e6


def _usd_short_rate_at_dates(precache, usd_factor, ref_date, comparison_dates, bridge=False):
    """-ln(DF(0,t))/t at each comparison date, for every path -- either
    read directly (Method A, date is on the native grid) or via state_at()
    (Method B, bridge interpolation to an off-grid date)."""
    model = precache.rate_models[usd_factor]
    cached = precache.rate_states[usd_factor]
    n_paths = cached.shape[0]
    sim_times = precache.sim_times

    out = np.zeros((len(comparison_dates), n_paths))
    for i, d in enumerate(comparison_dates):
        t = year_frac(ref_date, d)
        if bridge:
            state_at_t = model.state_at(cached, sim_times, t)   # (n_paths, n_state) -- mean-only bridge
        else:
            # Method A: d is (approximately) on the native daily grid --
            # find the matching index directly rather than re-interpolating
            idx = min(range(len(sim_times)), key=lambda k: abs(sim_times[k] - t))
            state_at_t = cached[:, idx, :]
        dfs = np.array([model.discount_factor(state_at_t[p], 0.0, t) for p in range(n_paths)])
        out[i, :] = -np.log(dfs) / t
    return out   # (n_comparison_dates, n_paths)


def _equity_level_at_dates(precache, eq_factor, usd_factor, ref_date, comparison_dates, bridge=False):
    model = precache.spot_models[eq_factor]
    cached = precache.spot_states[eq_factor]
    rate_model = precache.rate_models[usd_factor]
    rate_cached = precache.rate_states[usd_factor]
    n_paths = cached.shape[0]
    sim_times = precache.sim_times

    out = np.zeros((len(comparison_dates), n_paths))
    for i, d in enumerate(comparison_dates):
        t = year_frac(ref_date, d)
        if bridge:
            state_at_t = model.state_at(cached, sim_times, t)
            rate_state_at_t = rate_model.state_at(rate_cached, sim_times, t)
        else:
            idx = min(range(len(sim_times)), key=lambda k: abs(sim_times[k] - t))
            state_at_t = cached[:, idx, :]
            rate_state_at_t = rate_cached[:, idx, :]
        levels = model.level_at_batch(state_at_t, t, rate_state_at_t)
        out[i, :] = levels
    return out


def _summ_stats(arr_2d):
    """arr_2d: (n_dates, n_paths) -> per-date summary dict."""
    return {
        "mean": arr_2d.mean(axis=1).tolist(),
        "std": arr_2d.std(axis=1, ddof=1).tolist(),
        "q05": np.percentile(arr_2d, 5, axis=1).tolist(),
        "q50": np.percentile(arr_2d, 50, axis=1).tolist(),
        "q95": np.percentile(arr_2d, 95, axis=1).tolist(),
        "min": arr_2d.min(axis=1).tolist(),
        "max": arr_2d.max(axis=1).tolist(),
    }


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    book = build_sourced_book(ref_date=REF_DATE)
    sim = book["build_simulator"]()
    usd_factor = book["factors"].rates[0]
    eq_factor = next(e for e in book["factors"].equities if e.isin in book["market_data"]["equity_spot"])

    comparison_dates = _comparison_dates(REF_DATE, HORIZON_YEARS)
    print(f"Comparing at {len(comparison_dates)} calendar dates spanning {comparison_dates[0]} to {comparison_dates[-1]}")

    print(f"\n--- Method A: full daily-grid simulation ({N_PATHS:,} paths) ---")
    precache_a, dates_a, time_a, mem_a = method_a_full_grid(sim, book["market_state_for_correlation"], N_PATHS,
                                                            REF_DATE, HORIZON_YEARS, SEED)
    print(f"  {len(dates_a)} daily dates simulated directly. sim_time={time_a:.3f}s, peak_mem={mem_a:.1f}MB")

    print(f"\n--- Method B: coarse grid + Brownian-bridge interpolation ({N_PATHS:,} paths) ---")
    precache_b, dates_b, time_b, mem_b = method_b_bridge(sim, book["market_state_for_correlation"], N_PATHS,
                                                         REF_DATE, HORIZON_YEARS, SEED)
    print(f"  {len(dates_b)} coarse dates simulated directly ({len(dates_a)/len(dates_b):.1f}x fewer than Method A). "
          f"sim_time={time_b:.3f}s, peak_mem={mem_b:.1f}MB")

    print(f"\nRuntime ratio: Method A / Method B = {time_a/time_b:.2f}x  "
          f"(Method B is {(1 - time_b/time_a)*100:.1f}% faster)" if time_a > time_b else
          f"\nRuntime ratio: Method A / Method B = {time_a/time_b:.2f}x")

    print("\n--- Evaluating USD short rate at comparison dates ---")
    t0 = time.perf_counter()
    rate_a = _usd_short_rate_at_dates(precache_a, usd_factor, REF_DATE, comparison_dates, bridge=False)
    eval_time_a_rate = time.perf_counter() - t0
    t0 = time.perf_counter()
    rate_b = _usd_short_rate_at_dates(precache_b, usd_factor, REF_DATE, comparison_dates, bridge=True)
    eval_time_b_rate = time.perf_counter() - t0
    print(f"  Method A eval time: {eval_time_a_rate:.3f}s, Method B (bridge) eval time: {eval_time_b_rate:.3f}s")

    print("\n--- Evaluating equity level at comparison dates ---")
    eq_a = _equity_level_at_dates(precache_a, eq_factor, usd_factor, REF_DATE, comparison_dates, bridge=False)
    eq_b = _equity_level_at_dates(precache_b, eq_factor, usd_factor, REF_DATE, comparison_dates, bridge=True)

    stats_a_rate, stats_b_rate = _summ_stats(rate_a), _summ_stats(rate_b)
    stats_a_eq, stats_b_eq = _summ_stats(eq_a), _summ_stats(eq_b)

    # Since path 0 in A and path 0 in B are NOT the same realization (see
    # module docstring), MAE/RMSE "vs Method A" is computed on the
    # PER-DATE SUMMARY STATISTICS (mean, std, quantiles) -- comparing two
    # ensembles' distributional moments at each date, the statistically
    # correct comparison here, not a pointwise per-path difference.
    def _stat_errors(a, b, key):
        a_arr, b_arr = np.array(a[key]), np.array(b[key])
        return {
            "mae": float(np.mean(np.abs(a_arr - b_arr))),
            "rmse": float(np.sqrt(np.mean((a_arr - b_arr) ** 2))),
            "max_abs_diff": float(np.max(np.abs(a_arr - b_arr))),
            "corr": float(np.corrcoef(a_arr, b_arr)[0, 1]) if len(a_arr) > 1 else float("nan"),
        }

    print("\n=== USD short rate: Method A (full grid) vs Method B (bridge) -- per-date summary-stat errors ===")
    rate_errors = {}
    for key in ("mean", "std", "q05", "q50", "q95"):
        err = _stat_errors(stats_a_rate, stats_b_rate, key)
        rate_errors[key] = err
        print(f"  {key:>5s}: MAE={err['mae']*10000:.4f}bp  RMSE={err['rmse']*10000:.4f}bp  "
              f"max|diff|={err['max_abs_diff']*10000:.4f}bp  corr={err['corr']:.6f}")

    print("\n=== Equity level: Method A (full grid) vs Method B (bridge) -- per-date summary-stat errors ===")
    eq_errors = {}
    for key in ("mean", "std", "q05", "q50", "q95"):
        err = _stat_errors(stats_a_eq, stats_b_eq, key)
        eq_errors[key] = err
        print(f"  {key:>5s}: MAE={err['mae']:.6f}  RMSE={err['rmse']:.6f}  max|diff|={err['max_abs_diff']:.6f}  "
              f"corr={err['corr']:.6f}")

    print("\n=== Trade-off summary ===")
    time_saved_pct = (1 - time_b / time_a) * 100 if time_a else 0.0
    mem_saved_pct = (1 - mem_b / mem_a) * 100 if mem_a else 0.0
    print(f"  Simulation time saved (Method B vs A): {time_saved_pct:+.1f}%")
    print(f"  Peak memory saved (Method B vs A): {mem_saved_pct:+.1f}%")
    print(f"  USD rate mean-path RMSE (bridge vs full grid): {rate_errors['mean']['rmse']*10000:.4f} bp")
    print(f"  Equity mean-path RMSE (bridge vs full grid): {eq_errors['mean']['rmse']:.6f} "
          f"({eq_errors['mean']['rmse']/np.mean(stats_a_eq['mean'])*100:.4f}% of mean level)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "script": "bridge_vs_full_grid.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "n_paths": N_PATHS,
        "ref_date": REF_DATE.isoformat(),
        "horizon_years": HORIZON_YEARS,
        "n_daily_dates_method_a": len(dates_a),
        "n_coarse_dates_method_b": len(dates_b),
        "n_comparison_dates": len(comparison_dates),
        "caveat": "Method A and B use the SAME seed but DIFFERENT n_steps, so they consume the "
                  "RNG stream differently -- path i in A is NOT the same Brownian realization as "
                  "path i in B. This is a distributional/statistical comparison of the two "
                  "ensembles' moments at matched calendar dates, not a per-path equality check.",
        "runtime": {"method_a_sim_s": time_a, "method_b_sim_s": time_b,
                   "method_a_eval_s": eval_time_a_rate, "method_b_eval_s": eval_time_b_rate,
                   "time_saved_pct": time_saved_pct},
        "memory": {"method_a_peak_mb": mem_a, "method_b_peak_mb": mem_b, "mem_saved_pct": mem_saved_pct},
        "usd_rate_stats_method_a": stats_a_rate, "usd_rate_stats_method_b": stats_b_rate,
        "usd_rate_errors": rate_errors,
        "equity_stats_method_a": stats_a_eq, "equity_stats_method_b": stats_b_eq,
        "equity_errors": eq_errors,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                   "python_version": platform.python_version(), "cpu_count": os.cpu_count()},
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved reproducible results to {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
