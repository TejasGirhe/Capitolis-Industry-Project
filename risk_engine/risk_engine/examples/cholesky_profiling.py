"""
Section 4/5 of the simulation-framework deep-dive: instrumented timing
breakdown of JointSimulator.simulate() -- measures where time actually goes
(correlation-matrix build, PSD repair, Cholesky factorization, random-number
generation, the Z_indep @ chol.T transform, and each factor's own
simulate_paths() call), WITHOUT modifying the production simulate() method
itself (per the "do not modify production until baseline is benchmarked"
requirement) -- this script re-implements simulate()'s exact steps with
timers around each one, calling the SAME underlying methods
(_build_correlation_matrix, _nearest_psd, model.simulate_paths) the real
method calls, so the trace is a faithful line-for-line timing of production
logic, not an approximation of it.

    python risk_engine/examples/cholesky_profiling.py
"""
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import date

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.models._shared import year_frac
from risk_engine.pricing import price_curves

REF_DATE = date(2026, 8, 31)
PATH_COUNTS = [1000, 5000, 10000, 25000, 50000]
SEED = 123
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "cholesky_profiling_results.json")


def instrumented_simulate(sim, market_state, n_paths, horizon_dates, rng, ref_date):
    """Line-for-line re-implementation of JointSimulator.simulate(), timed
    at each stage. Calls the SAME private methods production uses
    (_driver_layout, _build_correlation_matrix) so this is a faithful
    trace, not a re-derivation of the logic."""
    timings = {}

    t0 = time.perf_counter()
    factor_order, slot_ranges, n_drivers = sim._driver_layout()
    timings["driver_layout"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    corr = sim._build_correlation_matrix(market_state, factor_order, slot_ranges, n_drivers)
    timings["build_correlation_matrix_and_psd_repair"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chol = np.linalg.cholesky(corr)
    timings["cholesky_factorization"] = time.perf_counter() - t0

    times = [0.0] + [year_frac(ref_date, d) for d in horizon_dates]
    n_steps = len(times) - 1

    t0 = time.perf_counter()
    z_indep = rng.standard_normal((n_paths, n_steps, n_drivers))
    timings["rng_standard_normal_draw"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    z_joint = z_indep @ chol.T
    timings["z_joint_matrix_multiply"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rate_states, rate_models = {}, {}
    for factor, model in sim._rate_factors:
        s, n = slot_ranges[factor]
        ext_z = z_joint[:, :, s:s + n]
        rate_states[factor] = model.simulate_paths(n_paths, horizon_dates, rng, external_z=ext_z)
        rate_models[factor] = model
    timings["rate_factor_simulate_paths"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    spot_states, spot_models = {}, {}
    for factor, model, _drift_factor, _foreign_factor in sim._spot_factors:
        s, n = slot_ranges[factor]
        ext_z = z_joint[:, :, s:s + n]
        spot_states[factor] = model.simulate_paths(n_paths, horizon_dates, rng, external_z=ext_z)
        spot_models[factor] = model
    timings["spot_factor_simulate_paths"] = time.perf_counter() - t0

    timings["n_drivers"] = n_drivers
    timings["n_paths"] = n_paths
    timings["n_steps"] = n_steps

    from risk_engine.simulation.joint import JointSimResult
    precache = JointSimResult(
        factor_order=factor_order, rate_states=rate_states, spot_states=spot_states,
        rate_models=rate_models, spot_models=spot_models,
        horizon_dates=list(horizon_dates), sim_times=[year_frac(ref_date, d) for d in horizon_dates],
        ref_date=ref_date, corr_chol=chol, slot_ranges=slot_ranges,
        drift_rate_factor_by_spot={f: df for f, _, df, _ in sim._spot_factors},
        foreign_rate_factor_by_spot={f: ff for f, _, _, ff in sim._spot_factors if ff is not None},
    )
    return timings, precache


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    t0 = time.perf_counter()
    book = build_sourced_book(ref_date=REF_DATE)
    sourcing_time = time.perf_counter() - t0
    print(f"  market-data sourcing (FRED/Yahoo/Bloomberg/OPRA) + base rate calibration: {sourcing_time:.2f}s "
          f"-- ONE-TIME cost per book load, NOT part of per-run simulation timing below")

    # build_simulator() re-calibrates the rate model closure (cheap -- LGM/GBM/FX
    # calibration is closed-form pillar bootstrapping, not iterative optimization)
    # plus builds every equity/FX CalibratedSpotModel -- timed separately from
    # sourcing to isolate "calibration" from "data fetch" per the requested breakdown.
    t0 = time.perf_counter()
    sim = book["build_simulator"]()
    calibration_time = time.perf_counter() - t0
    print(f"  model calibration (LGM2F_SV x2 currencies + GBM_SV x37 + FXGBM_SV, closed-form): {calibration_time:.4f}s")

    grid = book["grid"]

    print(f"\n{'n_paths':>8s}  {'driver_layout':>14s}  {'corr+psd':>10s}  {'cholesky':>10s}  "
          f"{'rng_draw':>10s}  {'z_matmul':>10s}  {'rate_sim':>10s}  {'spot_sim':>10s}  {'sim_TOTAL':>10s}  "
          f"{'pricing':>10s}  {'peak_MB':>9s}")

    results = []
    for n_paths in PATH_COUNTS:
        tracemalloc.start()
        rng = np.random.default_rng(SEED)
        timings, precache = instrumented_simulate(sim, book["market_state_for_correlation"], n_paths, grid.dates,
                                                   rng, REF_DATE)
        sim_total = sum(v for k, v in timings.items() if k not in ("n_drivers", "n_paths", "n_steps"))

        t0 = time.perf_counter()
        # n_workers=8, not None (os.cpu_count()=24 on this machine) -- a
        # 24-worker Windows multiprocessing Pool deadlocked indefinitely
        # under load (WinError 87 in spawn_main's OpenProcess call for a
        # subset of workers). See mc_convergence_analysis.py for the
        # incident details.
        price_curves(book["trades"], precache, book["regression_dates"],
                    equity_dividend_rates=book["market_data"]["equity_dividend_rates"], n_workers=8)
        pricing_time = time.perf_counter() - t0

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({"n_paths": n_paths, "timings": {k: v for k, v in timings.items()
                        if k not in ("n_drivers", "n_paths", "n_steps")},
                        "n_drivers": timings["n_drivers"], "n_steps": timings["n_steps"],
                        "sim_total_s": sim_total, "pricing_s": pricing_time, "peak_memory_mb": peak / 1e6})
        print(f"{n_paths:>8d}  {timings['driver_layout']*1000:>12.3f}ms  "
              f"{timings['build_correlation_matrix_and_psd_repair']*1000:>8.2f}ms  "
              f"{timings['cholesky_factorization']*1000:>8.3f}ms  "
              f"{timings['rng_standard_normal_draw']:>9.3f}s  "
              f"{timings['z_joint_matrix_multiply']:>9.3f}s  "
              f"{timings['rate_factor_simulate_paths']:>9.3f}s  "
              f"{timings['spot_factor_simulate_paths']:>9.3f}s  "
              f"{sim_total:>9.3f}s  {pricing_time:>9.3f}s  {peak/1e6:>8.1f}MB")

    print(f"\nn_drivers = {results[0]['n_drivers']}, n_steps = {results[0]['n_steps']}")
    print("\n=== Timing breakdown as % of (sim + pricing) TOTAL, per path count ===")
    for r in results:
        grand_total = r["sim_total_s"] + r["pricing_s"]
        t = r["timings"]
        print(f"  n={r['n_paths']:>6d}  (grand total incl. pricing = {grand_total:.3f}s):")
        for label, key in [("Cholesky factorization", "cholesky_factorization"),
                           ("Correlation-matrix build + PSD repair", "build_correlation_matrix_and_psd_repair"),
                           ("RNG (standard_normal draw)", "rng_standard_normal_draw"),
                           ("Correlated-shock matmul (Z @ L^T)", "z_joint_matrix_multiply"),
                           ("Rate factor path construction", "rate_factor_simulate_paths"),
                           ("Spot factor path construction", "spot_factor_simulate_paths")]:
            pct = t[key] / grand_total * 100 if grand_total else 0.0
            print(f"    {label:<42s} {pct:>6.3f}%   ({t[key]:.4f}s)")
        pricing_pct = r["pricing_s"] / grand_total * 100 if grand_total else 0.0
        print(f"    {'Pricing (price_curves)':<42s} {pricing_pct:>6.3f}%   ({r['pricing_s']:.4f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "script": "cholesky_profiling.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "ref_date": REF_DATE.isoformat(),
        "path_counts": PATH_COUNTS,
        "n_drivers": results[0]["n_drivers"],
        "n_steps": results[0]["n_steps"],
        "one_time_sourcing_s": sourcing_time,
        "one_time_calibration_s": calibration_time,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                   "python_version": platform.python_version(), "cpu_count": os.cpu_count()},
        "results": results,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved reproducible results to {RESULTS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
