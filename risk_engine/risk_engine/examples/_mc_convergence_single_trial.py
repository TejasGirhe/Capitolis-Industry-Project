"""
Single-trial worker for mc_convergence_analysis.py, run as an INDEPENDENT
OS PROCESS (not a thread/multiprocessing.Pool worker) so many trials can
run truly concurrently as separate processes, each with its own Python
interpreter, GIL, and (critically) its own SEPARATE small worker pool --
avoiding the nested-pool-of-pools pattern that would oversubscribe cores.

Each invocation sources the book itself (book state is NOT shared across
OS processes the way it is within a single Python process's loop) --
this trades the one-time ~5.5min sourcing cost paid ONCE per invocation
against true process-level parallelism across trials. With up to 20
concurrent invocations, sourcing itself would be run up to 20x redundantly
if done naively; instead each worker uses a LOCAL DISK CACHE of the
sourced book's raw inputs (via _sourced_book.py's own caching, see below)
so only the first process to run pays the live FRED/Yahoo fetch cost and
the rest reuse the cached fetch.

n_workers=1 for price_curves() in THIS script deliberately -- each of the
(up to 20) concurrent OS processes gets exactly 1 CPU's worth of pricing
work; the OS scheduler timeshares the 24 logical CPUs across however many
concurrent trial-processes are launched (<=20 here, by design, leaving
headroom below the 24-core WinError-87 threshold observed earlier).

Usage: python _mc_convergence_single_trial.py <n_paths> <seed> <output_json_path>
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

TRADE_IDS = ["BF_0003", "EQTRS_0001", "BTRS_0001"]
REF_DATE = date(2026, 8, 31)


def main():
    n_paths = int(sys.argv[1])
    seed = int(sys.argv[2])
    out_path = sys.argv[3]

    t0 = time.time()
    book = build_sourced_book(ref_date=REF_DATE)
    sourcing_time = time.time() - t0

    trades_present = [tid for tid in TRADE_IDS if tid in book["trades"]]
    analytic = {tid: _analytic_npv(book["trades"][tid], book) for tid in trades_present}

    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]
    sim = book["build_simulator"]()

    t0 = time.time()
    rng = np.random.default_rng(seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                            rng=rng, ref_date=book["ref_date"])
    # n_workers=1: this trial is ITSELF one of up to 20 concurrent OS
    # processes: giving it its own internal 8-worker pool (as the
    # sequential version did) would oversubscribe to up to 160 total
    # worker processes across 24 cores -- exactly the failure mode
    # (WinError 87 / Pool deadlock) already observed at just 24 workers
    # from a SINGLE pool. n_workers=1 means price_curves runs single-
    # threaded in-process; concurrency comes entirely from running many
    # trial-processes side by side, not from nesting pools.
    result = price_curves(book["trades"], precache, regression_dates,
                          equity_dividend_rates=equity_div, n_workers=1)
    sim_and_price_time = time.time() - t0

    eval_dates = {tid: _eval_date_for(book["trades"][tid]) for tid in trades_present}
    npvs = {tid: result.mean_npv0_by_trade(d)[tid] for tid, d in eval_dates.items()}

    counterparties = build_netting_hierarchy(book["trades"])
    profiles = compute_all_profiles(counterparties, result, anchors, book["ref_date"])
    book_total = profiles["BOOK_TOTAL"]
    max_ee = max(book_total.ee) if book_total.ee else 0.0
    mpe99 = book_total.mpe_99

    payload = {
        "n_paths": n_paths, "seed": seed,
        "analytic": analytic, "npvs": npvs,
        "max_ee": max_ee, "mpe99": mpe99,
        "sourcing_s": sourcing_time, "sim_and_price_s": sim_and_price_time,
        "pid": os.getpid(),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, out_path)
    print(f"[pid {os.getpid()}] n={n_paths} seed={seed} DONE  max_EE=${max_ee:,.2f}  MPE_99=${mpe99:,.2f}  "
          f"sourcing={sourcing_time:.1f}s sim+price={sim_and_price_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
