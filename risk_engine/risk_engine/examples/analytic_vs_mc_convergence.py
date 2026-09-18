"""
Analytic price vs. Monte Carlo price, across path counts, for ONE
representative trade of each product type in the book (bond forward,
equity TRS, bond TRS). Extends single_trade_convergence.py's BF_0003-only
study to all three product types, since all three have the SAME structural
property: their NPV's TRUE MEAN has an exact closed form, purely curve-
derived, with NO dependence on any factor's volatility --

  - BondForwardTrade: NPV = notional/face * DF(T) * (forward_clean(T) - strike)
    (bond.py's docstring / this project's own test_lgm_forward_matching.py)
  - EquityTRS: equity leg = shares * DF(end) * (S/DF(end) - basis) -- the
    total-return forward S/DF(end) grows at the risk-free rate REGARDLESS
    of equity volatility (100% dividend pass-through nets the vol-dependent
    drag out entirely -- equity_trs.py's own module docstring states this
    explicitly: "the dividend rate does not enter the price", and by the
    same martingale argument volatility does not enter the MEAN price
    either, only the variance of outcomes around it).
  - BondTRS: return leg uses forward_dirty(t), curve-derived -- bond_trs.py's
    own module docstring states this outright: "deterministic-curve forward
    prices are used for E[P_dirty(t)] ... this is the analytic benchmark
    it converges to."

So for all three, "the Monte Carlo mean NPV should converge to a known,
closed-form number as paths increase" is not a hopeful analogy from the
bond-forward case -- it is a property this project's own pricer modules
already document about themselves. This script makes that check concrete
and quantitative, across path counts, for one trade of each type.

EVAL_DATE per trade: NOT ref_date (every path is identical, deterministic,
zero-diffusion at ref_date -- see single_trade_convergence.py's docstring
for the bug this caused when first tried there). Uses that same trade's own
final settlement/maturity date instead, where genuine path-to-path
variation exists and the payoff has fully realized.

    python risk_engine/examples/analytic_vs_mc_convergence.py
"""
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

from capitolis_pricers.pricers.bond_forward import BondForwardTrade
from capitolis_pricers.pricers.equity_trs import EquityTRS
from capitolis_pricers.pricers.bond_trs import BondTRS
from risk_engine.pricing import price_curves
from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.examples.variance_reduction_comparison import _AnalyticMarketStub

PATH_COUNTS = [500, 1000, 2000, 5000, 10000, 20000]
REF_DATE = date(2026, 8, 24)
TRADE_IDS = ["BF_0003", "EQTRS_0001", "BTRS_0001"]


class _CurveOnlyMarketStub:
    """Like _AnalyticMarketStub but also serves equity spot / FX for the
    ANALYTIC (t=0, no-vol) benchmark -- EquityTRS/BondTRS need
    market.equity_spot(isin) and market.fx(...) in addition to
    market.discount(ccy), which _AnalyticMarketStub (built for
    BondForwardTrade alone) does not provide."""
    def __init__(self, usd_curve, ref_date, equity_spot, fx_rate=1.0, reporting_ccy="USD"):
        self._curve = usd_curve
        self.ref_date = ref_date
        self._equity_spot = equity_spot
        self._fx_rate = fx_rate
        self.reporting_ccy = reporting_ccy

    def discount(self, ccy):
        return self._curve

    def credit(self, issuer):
        return None

    def equity_spot(self, isin):
        return self._equity_spot[isin]

    def fx(self, base_ccy, quote_ccy):
        return 1.0 if base_ccy == quote_ccy else self._fx_rate


def _analytic_npv(trade, book):
    """The trade's analytic (t=0, curve-only, zero-vol) NPV -- the ground
    truth every method's Monte Carlo mean is compared against. Built from
    a market stub carrying ONLY t=0 curve/spot data, no simulated state at
    all, so this number has zero Monte Carlo noise by construction."""
    usd_curve = book["market_data"]["usd_curve"]
    equity_spot = book["market_data"]["equity_spot"]
    market = _CurveOnlyMarketStub(usd_curve, REF_DATE, equity_spot)
    return trade.npv(market, reporting=True)


def _eval_date_for(trade):
    """The trade's own final settlement/maturity date -- guaranteed to be a
    regression date (every trade type's cashflow-date list includes its own
    end/forward date, per simulation.grid.trade_cashflow_dates), and where
    the payoff has genuinely realized (real path-to-path variation exists,
    unlike ref_date -- see module docstring)."""
    if isinstance(trade, BondForwardTrade):
        return trade.forward_date
    return trade.end_date  # EquityTRS, BondTRS


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    book = build_sourced_book(ref_date=REF_DATE)
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    trades = {tid: book["trades"][tid] for tid in TRADE_IDS}
    eval_dates = {tid: _eval_date_for(t) for tid, t in trades.items()}
    analytic = {tid: _analytic_npv(t, book) for tid, t in trades.items()}

    print("\nAnalytic (ground-truth, zero-vol, curve-only) NPVs:")
    for tid in TRADE_IDS:
        print(f"  {tid:12s} ({type(trades[tid]).__name__:16s})  eval_date={eval_dates[tid]}  "
              f"analytic_npv={analytic[tid]:>16,.2f}")

    print(f"\n{'n_paths':>8s}  {'trade':>12s}  {'NPV_mc':>16s}  {'bias($)':>14s}  {'bias(%)':>9s}  {'time(s)':>8s}")
    results = {tid: [] for tid in TRADE_IDS}
    for n_paths in PATH_COUNTS:
        sim = book["build_simulator"]()
        rng = np.random.default_rng(42)
        t0 = time.time()
        precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                                 rng=rng, ref_date=REF_DATE)
        result = price_curves(trades, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=None)
        elapsed = time.time() - t0

        for tid in TRADE_IDS:
            npv_mc = result.mean_npv0_by_trade(eval_dates[tid])[tid]
            bias = npv_mc - analytic[tid]
            bias_pct = bias / analytic[tid] * 100 if analytic[tid] else float("nan")
            results[tid].append((n_paths, npv_mc, bias, bias_pct))
            print(f"{n_paths:>8d}  {tid:>12s}  {npv_mc:>16,.2f}  {bias:>14,.2f}  {bias_pct:>8.3f}%  {elapsed:>8.1f}")

    print("\n=== Convergence: |bias %| as n_paths increases, per trade ===")
    for tid in TRADE_IDS:
        print(f"\n  {tid} ({type(trades[tid]).__name__}):")
        for n_paths, npv_mc, bias, bias_pct in results[tid]:
            print(f"    {n_paths:>8d} paths:  |bias| = {abs(bias_pct):>7.3f}%   (${abs(bias):>14,.2f})")


if __name__ == "__main__":
    main()
