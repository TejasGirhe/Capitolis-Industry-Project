"""
MC convergence, broken out BY TRADE TYPE (Bond Forward, Equity TRS, Bond
TRS), comparing PLAIN Monte Carlo against ANTITHETIC VARIATES at several
path counts -- extends variance_reduction_comparison.py's book-level-only
comparison to report each trade type's OWN RMSE against its OWN analytic
reference (see analytic_vs_mc_convergence.py), since a book-level number
can hide one trade type converging well while another doesn't (exactly
the EQTRS_0001/BTRS_0001 bias finding from this session's earlier
mc_convergence_analysis.py run).

One representative trade per type is used (not every trade of that type),
chosen as the type's largest-notional trade so the RMSE numbers are on a
scale that matters -- reported per-trade AND normalized to % of that
trade's analytic NPV so trades of different sizes are comparable.

Control variates are NOT included here for equity/bond TRS -- no exact
analytic reference exists under stochastic vol for either (see
variance_reduction_comparison.py's module docstring for why), so a CV
correction for those trade types would itself be an approximation with no
way to validate its own correctness. Only plain vs. antithetic is
compared for all three types; the bond forward ALSO gets its already-
validated CV correction (reusing variance_reduction_comparison.py's
tested implementation directly, not a re-derivation) since it has a real
analytic reference to correct against.

    python risk_engine/examples/per_trade_type_convergence.py
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

from capitolis_pricers.pricers.bond_forward import BondForwardTrade
from capitolis_pricers.pricers.equity_trs import EquityTRS
from capitolis_pricers.pricers.bond_trs import BondTRS
from risk_engine.pricing import price_curves
from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.examples.analytic_vs_mc_convergence import _analytic_npv, _eval_date_for
from risk_engine.examples.variance_reduction_comparison import (
    _bond_forward_cv_at_date,
)
from risk_engine.greeks.sensitivities import notional_of

PATH_COUNTS = [1000, 2000, 5000, 10000, 20000]
SEEDS = [1, 2, 3]   # repeated trials per (method, path_count) for a genuine std, not single-run noise
N_WORKERS = 8
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "per_trade_type_convergence_checkpoint.json")


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


def _representative_trades(trades):
    """Largest-notional trade of each type -- one BondForwardTrade, one
    EquityTRS, one BondTRS."""
    by_type = {BondForwardTrade: [], EquityTRS: [], BondTRS: []}
    for tid, t in trades.items():
        for cls in by_type:
            if isinstance(t, cls):
                by_type[cls].append((tid, t))
    out = {}
    for cls, items in by_type.items():
        if not items:
            continue
        tid, t = max(items, key=lambda kv: abs(notional_of(kv[1])))
        out[cls.__name__] = (tid, t)
    return out


def _run_trial(book, n_paths, seed, antithetic):
    grid, regression_dates = book["grid"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]
    sim = book["build_simulator"]()

    rng = np.random.default_rng(seed)
    t0 = time.time()
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                            rng=rng, ref_date=book["ref_date"], antithetic=antithetic)
    result = price_curves(book["trades"], precache, regression_dates,
                          equity_dividend_rates=equity_div, n_workers=N_WORKERS)
    elapsed = time.time() - t0
    return precache, result, elapsed


def main():
    state = _load_checkpoint()
    print("Loading book (live cached market data)...")
    book = build_sourced_book()
    reps = _representative_trades(book["trades"])
    print(f"  representative trades: { {k: v[0] for k, v in reps.items()} }")

    analytic = {name: _analytic_npv(t, book) for name, (tid, t) in reps.items()}
    eval_dates = {name: _eval_date_for(t) for name, (tid, t) in reps.items()}
    print(f"  analytic NPVs: {analytic}")

    for n_paths in PATH_COUNTS:
        for seed in SEEDS:
            for method in ("plain", "antithetic"):
                key = f"{n_paths}_{seed}_{method}"
                if key in state["trials"]:
                    continue
                antithetic = method == "antithetic"
                precache, result, elapsed = _run_trial(book, n_paths, seed, antithetic)

                npvs = {}
                for name, (tid, trade) in reps.items():
                    d = eval_dates[name]
                    npvs[name] = float(result.mean_npv0_by_trade(d)[tid])

                # bond-forward CV correction (reuses the validated implementation)
                bf_name = "BondForwardTrade"
                cv_npv = None
                if bf_name in reps:
                    tid, trade = reps[bf_name]
                    d = eval_dates[bf_name]
                    forward_analytic, forward_mc = _bond_forward_cv_at_date(book, precache, trade, d)
                    n_paths_actual = next(iter(precache.rate_states.values())).shape[0]
                    npv_mc_per_path = np.array([
                        result.npv0[(p, d, tid)] for p in range(n_paths_actual)
                    ])
                    cov = np.cov(npv_mc_per_path, forward_mc)[0, 1]
                    var_fwd = np.var(forward_mc)
                    beta = cov / var_fwd if var_fwd > 0 else 0.0
                    npv_cv = npv_mc_per_path - beta * (forward_mc - forward_analytic)
                    cv_npv = float(npv_cv.mean())

                state["trials"][key] = {
                    "n_paths": n_paths, "seed": seed, "method": method,
                    "npvs": npvs, "cv_npv_bond_forward": cv_npv, "elapsed_s": elapsed,
                }
                _save_checkpoint(state)
                print(f"  n={n_paths:>6d} seed={seed} method={method:<10s} "
                      f"npvs={ {k: round(v) for k, v in npvs.items()} } "
                      f"cv_bf={round(cv_npv) if cv_npv is not None else None}  ({elapsed:.0f}s)")

    print("\n" + "=" * 100)
    print("PER-TRADE-TYPE RMSE vs. ANALYTIC, BY METHOD AND PATH COUNT")
    print("=" * 100)
    for name in reps:
        print(f"\n{name} (analytic NPV = {analytic[name]:,.2f}):")
        print(f"  {'n_paths':>8s}  {'method':<12s}  {'mean_NPV':>16s}  {'RMSE':>14s}  {'RMSE%':>8s}")
        for n_paths in PATH_COUNTS:
            for method in ("plain", "antithetic"):
                vals = [state["trials"][f"{n_paths}_{s}_{method}"]["npvs"][name]
                       for s in SEEDS if f"{n_paths}_{s}_{method}" in state["trials"]]
                if not vals:
                    continue
                vals = np.array(vals)
                errs = vals - analytic[name]
                rmse = float(np.sqrt(np.mean(errs ** 2)))
                rmse_pct = rmse / abs(analytic[name]) * 100 if analytic[name] else float("nan")
                print(f"  {n_paths:>8d}  {method:<12s}  {vals.mean():>16,.2f}  {rmse:>14,.2f}  {rmse_pct:>7.3f}%")
            if name == "BondForwardTrade":
                cv_vals = [state["trials"][f"{n_paths}_{s}_plain"]["cv_npv_bond_forward"]
                          for s in SEEDS if f"{n_paths}_{s}_plain" in state["trials"]
                          and state["trials"][f"{n_paths}_{s}_plain"]["cv_npv_bond_forward"] is not None]
                if cv_vals:
                    cv_vals = np.array(cv_vals)
                    errs = cv_vals - analytic[name]
                    rmse = float(np.sqrt(np.mean(errs ** 2)))
                    rmse_pct = rmse / abs(analytic[name]) * 100 if analytic[name] else float("nan")
                    print(f"  {n_paths:>8d}  {'control_var':<12s}  {cv_vals.mean():>16,.2f}  {rmse:>14,.2f}  {rmse_pct:>7.3f}%")

    out_path = os.path.join(RESULTS_DIR, "per_trade_type_convergence_results.json")
    with open(out_path, "w") as fh:
        json.dump({"representative_trades": {k: v[0] for k, v in reps.items()}, "analytic": analytic,
                  "path_counts": PATH_COUNTS, "seeds": SEEDS, "trials": state["trials"]}, fh, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()["PATH_COUNTS"] = [int(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 2:
        globals()["SEEDS"] = [int(x) for x in sys.argv[2].split(",")]
    main()
