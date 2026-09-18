"""
Rigorous MC convergence analysis: RMSE, standard error, and bias, computed
from REPEATED independent trials (5 different seeds per path count), not
inferred from a single run's within-sample spread -- the statistically
correct way to measure Monte Carlo sampling error.

Two parts, per the confirmed scope:

  1. PRICING (rigorous RMSE against an EXACT analytic reference): for
     BF_0003, EQTRS_0001, BTRS_0001, RMSE = sqrt(mean((mc_i - analytic)^2))
     over the 5 seeds at each path count, since a true, closed-form,
     zero-noise reference NPV exists (see analytic_vs_mc_convergence.py).

  2. BOOK-LEVEL EXPOSURE (max_EE, MPE_99 -- no exact analytic reference
     exists for a netted portfolio's tail exposure): RMSE computed against
     the MEAN of the largest path count's 5 seeds (10,000 paths) as a PROXY
     for truth, explicitly labeled as such throughout -- this is NOT a
     rigorous error metric the way (1) is, since the "truth" itself carries
     its own (smaller, but nonzero) sampling error, and the 10,000-path
     trials are NOT independent of the RMSE-vs-truth comparison at n=10,000
     (that row's RMSE is computed against a mean that includes its own
     values -- flagged explicitly in the printed output). Reported alongside
     the empirical standard error (std across the 5 seeds) at each path
     count, which IS rigorous regardless of whether a true reference exists.
     (No separate 20,000-path proxy-truth pass is run -- skipped per request
     to keep the sweep to PATH_COUNTS only.)

Market data is sourced ONCE and the same book/simulator reused across
every (path_count, seed) combination -- only the RNG seed changes between
runs, avoiding 25x the ~7-11 minute Bloomberg/OPRA sourcing cost.

    python risk_engine/examples/mc_convergence_analysis.py
"""
import json
import os
import sys
import time
from datetime import date

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
from risk_engine.examples.analytic_vs_mc_convergence import _analytic_npv, _eval_date_for

PATH_COUNTS = [2000, 3000, 5000, 10000]
TRUTH_PATH_COUNT = max(PATH_COUNTS)   # proxy truth = mean of the largest path count's own 5 seeds (20K pass skipped)
SEEDS = [1, 2, 3, 4, 5]
TRADE_IDS = ["BF_0003", "EQTRS_0001", "BTRS_0001"]
REF_DATE = date(2026, 8, 31)

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results", "mc_convergence_checkpoint.json")


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as fh:
            return json.load(fh)
    return {"analytic": None, "truth": None, "trials": {}}


def _save_checkpoint(state):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)   # atomic on both POSIX and Windows -- never left half-written


def _trial_key(n_paths, seed):
    return f"{n_paths}_{seed}"


def _run_once(book, n_paths, seed):
    """One simulate + price_curves + exposure-profile pass. Returns
    (per_trade_npv: {tid: npv}, book_total_max_ee, book_total_mpe99)."""
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]
    sim = book["build_simulator"]()

    rng = np.random.default_rng(seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                            rng=rng, ref_date=book["ref_date"])
    # n_workers capped at 8, NOT os.cpu_count() (24 on this machine): at
    # high path counts, spawning 24 Windows multiprocessing workers hit
    # WinError 87 (OpenProcess on a subset of workers fails during
    # spawn_main's handle duplication) and left the Pool deadlocked
    # indefinitely (observed: 24+ hours hung, near-zero cumulative CPU
    # time across workers). Confirmed via traceback during the incident.
    result = price_curves(book["trades"], precache, regression_dates,
                          equity_dividend_rates=equity_div, n_workers=8)

    eval_dates = {tid: _eval_date_for(book["trades"][tid]) for tid in TRADE_IDS if tid in book["trades"]}
    npvs = {tid: result.mean_npv0_by_trade(d)[tid] for tid, d in eval_dates.items()}

    counterparties = build_netting_hierarchy(book["trades"])
    profiles = compute_all_profiles(counterparties, result, anchors, book["ref_date"])
    book_total = profiles["BOOK_TOTAL"]
    max_ee = max(book_total.ee) if book_total.ee else 0.0
    mpe99 = book_total.mpe_99
    return npvs, max_ee, mpe99


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def main():
    state = _load_checkpoint()
    resuming = state["analytic"] is not None
    print(f"Loading book ONCE (ref date {REF_DATE}) -- reused across all {len(PATH_COUNTS)*len(SEEDS)} trials "
          f"(path counts: {PATH_COUNTS}, no 20K proxy-truth pass)..."
          + (" [RESUMING from checkpoint]" if resuming else ""))
    book = build_sourced_book(ref_date=REF_DATE)
    trades_present = [tid for tid in TRADE_IDS if tid in book["trades"]]

    if state["analytic"] is None:
        analytic = {tid: _analytic_npv(book["trades"][tid], book) for tid in trades_present}
        state["analytic"] = analytic
        _save_checkpoint(state)
    else:
        analytic = state["analytic"]
    print("Analytic (exact) NPVs:", {tid: round(v, 2) for tid, v in analytic.items()})

    print(f"\nNo separate 20,000-path proxy-truth pass -- proxy truth will be the MEAN of the "
          f"{TRUTH_PATH_COUNT:,}-path trials' own 5 seeds (computed after those trials run below).")

    # ---- repeated trials at each path count ----
    npv_trials = {tid: {n: [] for n in PATH_COUNTS} for tid in trades_present}
    max_ee_trials = {n: [] for n in PATH_COUNTS}
    mpe99_trials = {n: [] for n in PATH_COUNTS}

    single_step = os.environ.get("MC_CONVERGENCE_SINGLE_STEP") == "1"
    ran_one_new_trial = False

    for n_paths in PATH_COUNTS:
        for seed in SEEDS:
            key = _trial_key(n_paths, seed)
            if key in state["trials"]:
                cached = state["trials"][key]
                npvs, max_ee, mpe99 = cached["npvs"], cached["max_ee"], cached["mpe99"]
                print(f"  n={n_paths:>6d}  seed={seed}  max_EE=${max_ee:>14,.2f}  MPE_99=${mpe99:>14,.2f}  "
                      f"[FROM CHECKPOINT]")
            else:
                t0 = time.time()
                npvs, max_ee, mpe99 = _run_once(book, n_paths, seed)
                elapsed = time.time() - t0
                state["trials"][key] = {"npvs": npvs, "max_ee": max_ee, "mpe99": mpe99, "elapsed_s": elapsed}
                _save_checkpoint(state)
                print(f"  n={n_paths:>6d}  seed={seed}  max_EE=${max_ee:>14,.2f}  MPE_99=${mpe99:>14,.2f}  "
                      f"({elapsed:.0f}s)")
                ran_one_new_trial = True
            for tid in trades_present:
                npv_trials[tid][n_paths].append(npvs[tid])
            max_ee_trials[n_paths].append(max_ee)
            mpe99_trials[n_paths].append(mpe99)
            if single_step and ran_one_new_trial:
                print(f"\n[MC_CONVERGENCE_SINGLE_STEP=1] Stopping after this one newly-run trial "
                      f"({len(state['trials'])}/{len(PATH_COUNTS)*len(SEEDS)} total done). "
                      f"Re-run the same command to do the next one.")
                return

    # ================================================================ PART 1: PRICING RMSE
    print("\n" + "=" * 78)
    print("PART 1 -- Pricing RMSE vs. EXACT analytic reference (rigorous)")
    print("=" * 78)
    for tid in trades_present:
        print(f"\n{tid} (analytic = ${analytic[tid]:,.2f}):")
        print(f"  {'n_paths':>8s}  {'mean_NPV':>16s}  {'bias':>14s}  {'bias%':>8s}  {'std(5 seeds)':>14s}  {'RMSE':>14s}  {'RMSE%':>8s}")
        for n_paths in PATH_COUNTS:
            vals = np.array(npv_trials[tid][n_paths])
            mean_npv = vals.mean()
            bias = mean_npv - analytic[tid]
            bias_pct = bias / analytic[tid] * 100 if analytic[tid] else float("nan")
            std = vals.std(ddof=1)
            errs = vals - analytic[tid]
            rmse_val = rmse(errs)
            rmse_pct = rmse_val / abs(analytic[tid]) * 100 if analytic[tid] else float("nan")
            print(f"  {n_paths:>8d}  {mean_npv:>16,.2f}  {bias:>14,.2f}  {bias_pct:>7.3f}%  {std:>14,.2f}  "
                  f"{rmse_val:>14,.2f}  {rmse_pct:>7.3f}%")

    # ================================================================ PART 2: BOOK-LEVEL
    print("\n" + "=" * 78)
    print(f"PART 2 -- Book-level exposure: RMSE vs. mean-of-{TRUTH_PATH_COUNT:,}-path-seeds PROXY truth "
          f"(NOT a true reference; NOT independent at n={TRUTH_PATH_COUNT:,} -- see note below)")
    print("=" * 78)
    for label, trials in [("max_EE", max_ee_trials), ("MPE_99", mpe99_trials)]:
        truth = float(np.mean(trials[TRUTH_PATH_COUNT]))
        print(f"\n{label} (proxy truth = mean of {TRUTH_PATH_COUNT:,}-path seeds = ${truth:,.2f}):")
        print(f"  {'n_paths':>8s}  {'mean':>16s}  {'std(5 seeds)':>14s}  {'SE=std/sqrt(5)':>16s}  {'RMSE vs proxy':>16s}  {'RMSE%':>8s}")
        for n_paths in PATH_COUNTS:
            vals = np.array(trials[n_paths])
            mean_val = vals.mean()
            std = vals.std(ddof=1)
            se = std / np.sqrt(len(vals))
            errs = vals - truth
            rmse_val = rmse(errs)
            rmse_pct = rmse_val / abs(truth) * 100 if truth else float("nan")
            flag = "  <- NOT independent (truth derived from these same values)" if n_paths == TRUTH_PATH_COUNT else ""
            print(f"  {n_paths:>8d}  {mean_val:>16,.2f}  {std:>14,.2f}  {se:>16,.2f}  {rmse_val:>16,.2f}  "
                  f"{rmse_pct:>7.2f}%{flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
