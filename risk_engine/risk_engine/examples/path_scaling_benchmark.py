"""
Section 3 of the simulation-framework deep-dive: path-count scaling
benchmark -- total runtime AND runtime-per-path at 1K/5K/10K/25K/50K/100K
paths (100K skipped if infeasible on this machine within a reasonable
time budget -- see MAX_FEASIBLE_TIME_S below), for the CURRENT (baseline,
unmodified) simulate() + price_curves() pipeline. This is the baseline
this section's scaling numbers are compared against BEFORE any Cholesky
optimization is attempted (per the "establish baseline first" requirement).

    python risk_engine/examples/path_scaling_benchmark.py
"""
import json
import os
import platform
import sys
import time
from datetime import date

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.pricing import price_curves

REF_DATE = date(2026, 8, 31)
PATH_COUNTS = [1000, 5000, 10000, 25000, 50000, 100000]
SEED = 123
MAX_FEASIBLE_TIME_S = 1800   # skip a path count if the PREVIOUS one alone took longer than this
                              # (extrapolating that the next, ~2x-5x larger count would take too
                              # long for this analysis session) -- reported explicitly, not silently
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "path_scaling_benchmark_results.json")


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    t0 = time.perf_counter()
    book = build_sourced_book(ref_date=REF_DATE)
    sourcing_time = time.perf_counter() - t0
    print(f"  sourcing + base calibration: {sourcing_time:.2f}s (one-time, excluded from per-path-count timings below)")

    sim = book["build_simulator"]()
    grid = book["grid"]

    print(f"\n{'n_paths':>8s}  {'sim_time_s':>11s}  {'price_time_s':>13s}  {'total_s':>9s}  "
          f"{'total_us_per_path':>18s}  {'sim_us_per_path':>16s}  {'price_us_per_path':>18s}")

    results = []
    skipped = []
    prev_total = 0.0
    for n_paths in PATH_COUNTS:
        if prev_total > MAX_FEASIBLE_TIME_S:
            print(f"  SKIPPING n={n_paths:,}: previous path count took {prev_total:.0f}s "
                  f"(> {MAX_FEASIBLE_TIME_S}s budget) -- extrapolated to be infeasible in this session")
            skipped.append(n_paths)
            continue

        rng = np.random.default_rng(SEED)
        t0 = time.perf_counter()
        precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                                rng=rng, ref_date=REF_DATE)
        sim_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        # n_workers=8, not None (os.cpu_count()=24 on this machine) -- a
        # 24-worker Windows multiprocessing Pool deadlocked indefinitely
        # under load (WinError 87 in spawn_main's OpenProcess call for a
        # subset of workers). See mc_convergence_analysis.py for the
        # incident details.
        price_curves(book["trades"], precache, book["regression_dates"],
                    equity_dividend_rates=book["market_data"]["equity_dividend_rates"], n_workers=8)
        price_time = time.perf_counter() - t0

        total = sim_time + price_time
        prev_total = total
        us_per_path_total = total / n_paths * 1e6
        us_per_path_sim = sim_time / n_paths * 1e6
        us_per_path_price = price_time / n_paths * 1e6
        results.append({"n_paths": n_paths, "sim_time_s": sim_time, "price_time_s": price_time,
                        "total_s": total, "us_per_path_total": us_per_path_total,
                        "us_per_path_sim": us_per_path_sim, "us_per_path_price": us_per_path_price})
        print(f"{n_paths:>8d}  {sim_time:>11.3f}  {price_time:>13.3f}  {total:>9.3f}  "
              f"{us_per_path_total:>18.2f}  {us_per_path_sim:>16.2f}  {us_per_path_price:>18.2f}")

    print("\n=== Scaling behavior ===")
    if len(results) >= 2:
        for i in range(1, len(results)):
            r0, r1 = results[i - 1], results[i]
            path_ratio = r1["n_paths"] / r0["n_paths"]
            time_ratio = r1["total_s"] / r0["total_s"] if r0["total_s"] else float("nan")
            scaling = "sub-linear" if time_ratio < path_ratio else ("linear" if abs(time_ratio - path_ratio) < 0.15 * path_ratio else "super-linear")
            print(f"  {r0['n_paths']:>6d} -> {r1['n_paths']:>6d}: paths x{path_ratio:.1f}, time x{time_ratio:.2f} ({scaling})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "script": "path_scaling_benchmark.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "ref_date": REF_DATE.isoformat(),
        "requested_path_counts": PATH_COUNTS,
        "skipped_path_counts": skipped,
        "max_feasible_time_s_budget": MAX_FEASIBLE_TIME_S,
        "one_time_sourcing_s": sourcing_time,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                   "python_version": platform.python_version(), "cpu_count": os.cpu_count()},
        "results": results,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved reproducible results to {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
