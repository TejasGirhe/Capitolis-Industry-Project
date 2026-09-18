"""
Single-trade convergence & accuracy study: all four techniques (plain
Monte Carlo, mixed-RV shocks, antithetic variates, control variates) on
ONE trade -- BF_0003, the $500M bond forward already identified (in the
book-level sweeps) as dominating this book's Monte Carlo variance. Isolating
one trade removes cross-trade netting noise, so each technique's effect can
be read cleanly against a trade that ALSO has an exact analytic reference
price -- letting us report genuine ACCURACY (bias vs. the known-correct
answer), not just run-to-run stability, which the book-level sweeps could
not do (no book-level analytic reference exists).

Reuses the exact same simulation/pricing machinery as
mixed_rv_comparison.py and variance_reduction_comparison.py -- same book,
same JointSimulator, same shock_sampler/antithetic hooks on
JointSimulator.simulate(), same CV construction validated there (see that
module's docstring for why an EARLIER, buggy version of the CV logic is
NOT reused here -- the corrected _bond_forward_cv_at_date /
_cv_corrected_curve_result functions are imported directly).

ACCURACY (this script's addition beyond the book-level sweeps): BF_0003's
NPV at ref_date is available in closed form (bond.forward_clean, curve-
derived) with NO Monte Carlo noise -- this is the "ground truth" every
method's mean_npv0_by_trade estimate is compared against, reported as
both raw dollar bias and percentage bias.

    python risk_engine/examples/single_trade_convergence.py
"""
import os
import sys
import time
from datetime import date

import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.pricing import price_curves
from risk_engine.netting import build_netting_hierarchy
from risk_engine.exposure import compute_all_profiles
from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.examples.mixed_rv_comparison import mixed_shock_sampler
from risk_engine.examples.variance_reduction_comparison import (
    _bond_forward_cv_at_date, _cv_corrected_curve_result, _AnalyticMarketStub,
)

PATH_COUNTS = [500, 1000, 2000, 5000, 10000, 20000]
REF_DATE = date(2026, 8, 24)
TRADE_ID = "BF_0003"


def _run(book, trade, n_paths, shock_sampler=None, antithetic=False, seed=42):
    ref_date = book["ref_date"]
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    # Simulate the WHOLE BOOK's joint factors (realistic correlation
    # context -- BF_0003 discounts off the same USD curve every other
    # trade does), but price/report ONLY this one trade -- same idea as
    # price_curves_per_trade, applied to a single trade of interest rather
    # than every trade independently.
    sim = book["build_simulator"]()
    rng = np.random.default_rng(seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                             rng=rng, ref_date=ref_date, shock_sampler=shock_sampler, antithetic=antithetic)
    result = price_curves({TRADE_ID: trade}, precache, regression_dates,
                           equity_dividend_rates=equity_div, n_workers=None)
    counterparties = build_netting_hierarchy({TRADE_ID: trade})
    profiles = compute_all_profiles(counterparties, result, anchors, ref_date)
    return profiles["BOOK_TOTAL"], precache, result


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    book = build_sourced_book(ref_date=REF_DATE)
    trade = book["trades"][TRADE_ID]
    usd_curve = book["market_data"]["usd_curve"]

    # Ground truth is evaluated at the trade's OWN forward_date -- NOT at
    # ref_date. At ref_date (t=0) every simulated path shares the exact
    # same deterministic starting state (zero elapsed time, zero
    # accumulated diffusion), so mean_npv0_by_trade(ref_date) returns the
    # SAME noiseless value on every single path regardless of method -- a
    # real bug caught during smoke-testing here (every method showed
    # EXACTLY 0.000% bias at every path count, which is not statistically
    # possible for a genuine Monte Carlo estimator; the "ground truth" test
    # point had no simulated randomness in it at all).
    #
    # forward_date is guaranteed to be a regression date (BondForwardTrade's
    # only cashflow date, per simulation.grid.trade_cashflow_dates), and
    # price_curves' maturity guard keeps a trade live THROUGH its own
    # maturity (d <= maturities[tid]), so it is genuinely priced there --
    # by forward_date the payoff has fully realized, so this also amounts
    # to a settlement-date pricing accuracy check (NPV should converge to
    # the deterministic notional * (forward_clean(T) - strike) payoff as
    # simulated state approaches its known terminal value).
    EVAL_DATE = trade.forward_date

    forward_analytic_eval = trade.forward_clean_price(_AnalyticMarketStub(usd_curve, EVAL_DATE))
    df_eval = usd_curve.discount(trade.forward_date) / usd_curve.discount(EVAL_DATE)  # = 1.0 at EVAL_DATE=forward_date
    npv_analytic = (trade.notional / trade.bond.face) * df_eval * (forward_analytic_eval - trade.strike_clean)
    npv_analytic = npv_analytic if trade.position == "long" else -npv_analytic
    print(f"Trade {TRADE_ID}: notional={trade.notional:,.0f}, forward_date={trade.forward_date}, "
          f"strike={trade.strike_clean}")
    print(f"Ground-truth evaluation date: {EVAL_DATE} (ref_date={REF_DATE}, forward_date={trade.forward_date})")
    print(f"Analytic (ground-truth) NPV at {EVAL_DATE}: {npv_analytic:,.2f}\n")

    print(f"{'n_paths':>8s}  {'method':>11s}  {'NPV_mc':>16s}  {'bias($)':>14s}  {'bias(%)':>9s}  {'time(s)':>8s}")
    results = {}
    for n_paths in PATH_COUNTS:
        methods = [
            ("plain", dict()),
            ("mixed", dict(shock_sampler=mixed_shock_sampler)),
            ("antithetic", dict(antithetic=True)),
        ]
        plain_precache = plain_result = None
        for method_name, kwargs in methods:
            t0 = time.time()
            profile, precache, result = _run(book, trade, n_paths, **kwargs)
            elapsed = time.time() - t0
            if method_name == "plain":
                plain_precache, plain_result = precache, result
            npv_mc = result.mean_npv0_by_trade(EVAL_DATE)[TRADE_ID]
            bias = npv_mc - npv_analytic
            bias_pct = bias / npv_analytic * 100 if npv_analytic else float("nan")
            results[(n_paths, method_name)] = (npv_mc, bias, bias_pct, profile)
            print(f"{n_paths:>8d}  {method_name:>11s}  {npv_mc:>16,.2f}  {bias:>14,.2f}  {bias_pct:>8.3f}%  {elapsed:>8.1f}")

        # -- control variate: reuses the PLAIN run's precache/result above,
        # no re-simulation needed --
        t0 = time.time()
        cv_result = _cv_corrected_curve_result(book, plain_precache, plain_result, {TRADE_ID: trade})
        npv_mc_cv = cv_result.mean_npv0_by_trade(EVAL_DATE)[TRADE_ID]
        bias_cv = npv_mc_cv - npv_analytic
        bias_pct_cv = bias_cv / npv_analytic * 100 if npv_analytic else float("nan")
        counterparties = build_netting_hierarchy({TRADE_ID: trade})
        cv_profile = compute_all_profiles(counterparties, cv_result, book["anchors"], book["ref_date"])["BOOK_TOTAL"]
        elapsed = time.time() - t0
        results[(n_paths, "cv")] = (npv_mc_cv, bias_cv, bias_pct_cv, cv_profile)
        print(f"{n_paths:>8d}  {'cv':>11s}  {npv_mc_cv:>16,.2f}  {bias_cv:>14,.2f}  {bias_pct_cv:>8.3f}%  {elapsed:>8.1f}")

    print("\n=== Accuracy summary: |bias| relative to the analytic ground truth ===")
    print(f"{'n_paths':>8s}  {'plain':>10s}  {'mixed':>10s}  {'antithetic':>10s}  {'cv':>10s}   (all in |bias %|)")
    for n_paths in PATH_COUNTS:
        row = [abs(results[(n_paths, m)][2]) for m in ("plain", "mixed", "antithetic", "cv")]
        print(f"{n_paths:>8d}  {row[0]:>9.3f}%  {row[1]:>9.3f}%  {row[2]:>9.3f}%  {row[3]:>9.3f}%")

    print("\n=== Convergence: max_EE run-to-run stability (this trade alone) ===")
    print(f"{'n_paths':>8s}  {'method':>11s}  {'max_EE':>14s}  {'|delta %|':>10s}")
    for method in ("plain", "mixed", "antithetic", "cv"):
        prev = None
        for n_paths in PATH_COUNTS:
            profile = results[(n_paths, method)][3]
            max_ee = max(profile.ee) if profile.ee else 0.0
            if prev is None:
                print(f"{n_paths:>8d}  {method:>11s}  {max_ee:>14,.2f}  {'--':>10s}")
            else:
                pct = abs(max_ee - prev) / prev * 100 if prev else float("nan")
                print(f"{n_paths:>8d}  {method:>11s}  {max_ee:>14,.2f}  {pct:>9.2f}%")
            prev = max_ee


if __name__ == "__main__":
    main()
