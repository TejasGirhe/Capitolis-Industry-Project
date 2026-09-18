"""
Decides how many Monte Carlo paths are enough for a production run, by
directly trading off wall-clock time against the STABILITY of the book-
level exposure metrics a production report actually reports: EE (max),
MPE_95, MPE_99.

Method: run the full book-level simulate+price+exposure pipeline several
times at each candidate path count, each with a different seed, and report
per path count:
    - median (not mean -- robust to the occasional slow/fast outlier
      trial) wall-clock time
    - median max_EE, MPE_95, MPE_99 across the repeated trials
    - relative spread (IQR / median) of each metric across trials -- the
      practical stand-in for "how much would this number move if I re-ran
      this with a different seed", which is what actually matters for a
      production sign-off, more directly than an RMSE-vs-proxy-truth
      metric requiring a separate "truth" run
    - time cost of the NEXT step up in path count, so the marginal cost of
      more precision is explicit, not just the absolute numbers

Uses the SAME live-data-cached book as the rest of this session's
production run (benchmark_results/_raw_data_cache) -- no re-fetch.

    python risk_engine/examples/production_path_count_analysis.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.pricing import price_curves
from risk_engine.netting import build_netting_hierarchy
from risk_engine.exposure import compute_all_profiles
from risk_engine.examples._sourced_book import build_sourced_book

PATH_COUNTS = [1000, 2000, 5000, 10000, 20000]
SEEDS = [1, 2, 3, 4, 5]
N_WORKERS = 8   # capped, not os.cpu_count() -- see mc_convergence_analysis.py's incident note
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "production_path_count_checkpoint.json")


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as fh:
            return json.load(fh)
    return {"trials": {}}


def _save_checkpoint(state):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)


def _run_once(book, n_paths, seed):
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]
    sim = book["build_simulator"]()

    t0 = time.time()
    rng = np.random.default_rng(seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                            rng=rng, ref_date=book["ref_date"])
    result = price_curves(book["trades"], precache, regression_dates,
                          equity_dividend_rates=equity_div, n_workers=N_WORKERS)
    elapsed = time.time() - t0

    counterparties = build_netting_hierarchy(book["trades"])
    profiles = compute_all_profiles(counterparties, result, anchors, book["ref_date"])
    book_total = profiles["BOOK_TOTAL"]
    max_ee = max(book_total.ee) if book_total.ee else 0.0
    return {"max_ee": max_ee, "mpe_95": book_total.mpe_95, "mpe_99": book_total.mpe_99, "elapsed_s": elapsed}


def _iqr_ratio(vals):
    """IQR / median -- a relative-spread measure that's robust to a small
    number of trials (unlike a coefficient-of-variation using std, which is
    noisy at n=5) and directly answers 'how much would this number move if
    I reran with a different seed', in percentage-of-the-number terms."""
    vals = np.array(vals)
    med = np.median(vals)
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = q75 - q25
    return (iqr / abs(med) * 100) if med else float("nan")


def main():
    state = _load_checkpoint()
    print("Loading book (live cached market data)...")
    book = build_sourced_book()
    print(f"  ref_date={book['ref_date']}  trades={len(book['trades'])}")

    for n_paths in PATH_COUNTS:
        for seed in SEEDS:
            key = f"{n_paths}_{seed}"
            if key in state["trials"]:
                r = state["trials"][key]
                print(f"  n={n_paths:>6d}  seed={seed}  max_EE=${r['max_ee']:>12,.0f}  "
                      f"MPE_95=${r['mpe_95']:>12,.0f}  MPE_99=${r['mpe_99']:>12,.0f}  "
                      f"({r['elapsed_s']:.0f}s)  [checkpoint]")
                continue
            r = _run_once(book, n_paths, seed)
            state["trials"][key] = r
            _save_checkpoint(state)
            print(f"  n={n_paths:>6d}  seed={seed}  max_EE=${r['max_ee']:>12,.0f}  "
                  f"MPE_95=${r['mpe_95']:>12,.0f}  MPE_99=${r['mpe_99']:>12,.0f}  "
                  f"({r['elapsed_s']:.0f}s)")

    print("\n" + "=" * 100)
    print("PRODUCTION PATH-COUNT DECISION TABLE")
    print("=" * 100)
    header = (f"{'n_paths':>8s}  {'median_time_s':>14s}  {'median_max_EE':>16s}  {'EE_spread%':>11s}  "
              f"{'median_MPE95':>16s}  {'MPE95_spread%':>14s}  {'median_MPE99':>16s}  {'MPE99_spread%':>14s}")
    print(header)
    rows = []
    prev_time = None
    for n_paths in PATH_COUNTS:
        trials = [state["trials"][f"{n_paths}_{s}"] for s in SEEDS if f"{n_paths}_{s}" in state["trials"]]
        if not trials:
            continue
        times = [t["elapsed_s"] for t in trials]
        ees = [t["max_ee"] for t in trials]
        m95 = [t["mpe_95"] for t in trials]
        m99 = [t["mpe_99"] for t in trials]
        med_time = float(np.median(times))
        row = {
            "n_paths": n_paths, "median_time_s": med_time,
            "median_max_ee": float(np.median(ees)), "ee_spread_pct": _iqr_ratio(ees),
            "median_mpe_95": float(np.median(m95)), "mpe95_spread_pct": _iqr_ratio(m95),
            "median_mpe_99": float(np.median(m99)), "mpe99_spread_pct": _iqr_ratio(m99),
            "time_ratio_vs_prev": (med_time / prev_time) if prev_time else None,
        }
        rows.append(row)
        prev_time = med_time
        print(f"{n_paths:>8d}  {med_time:>14.1f}  {row['median_max_ee']:>16,.0f}  {row['ee_spread_pct']:>10.2f}%  "
              f"{row['median_mpe_95']:>16,.0f}  {row['mpe95_spread_pct']:>13.2f}%  "
              f"{row['median_mpe_99']:>16,.0f}  {row['mpe99_spread_pct']:>13.2f}%")

    print("\n" + "=" * 100)
    print("MARGINAL COST OF MORE PRECISION")
    print("=" * 100)
    for i in range(1, len(rows)):
        r0, r1 = rows[i - 1], rows[i]
        path_ratio = r1["n_paths"] / r0["n_paths"]
        time_ratio = r1["median_time_s"] / r0["median_time_s"] if r0["median_time_s"] else float("nan")
        ee_spread_reduction = r0["ee_spread_pct"] - r1["ee_spread_pct"]
        mpe99_spread_reduction = r0["mpe99_spread_pct"] - r1["mpe99_spread_pct"]
        print(f"  {r0['n_paths']:>6d} -> {r1['n_paths']:>6d}: "
              f"time x{time_ratio:.2f} ({r1['median_time_s']-r0['median_time_s']:+.0f}s), "
              f"EE spread {r0['ee_spread_pct']:.2f}%->{r1['ee_spread_pct']:.2f}% "
              f"({ee_spread_reduction:+.2f}pp), "
              f"MPE_99 spread {r0['mpe99_spread_pct']:.2f}%->{r1['mpe99_spread_pct']:.2f}% "
              f"({mpe99_spread_reduction:+.2f}pp)")

    payload = {"path_counts": PATH_COUNTS, "seeds": SEEDS, "n_workers": N_WORKERS,
              "ref_date": str(book["ref_date"]), "decision_table": rows}
    out_path = os.path.join(RESULTS_DIR, "production_path_count_decision.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved decision table to {out_path}")


if __name__ == "__main__":
    main()
