"""
EXPERIMENTAL comparison: plain Monte Carlo vs. two variance-reduction
techniques -- antithetic variates (JointSimulator.simulate(antithetic=True),
see its docstring in simulation/joint.py) and control variates (CV, applied
only to bond-forward trades, where an exact analytic forward price exists;
see module docstring below for why equity/bond TRS are excluded) -- at the
SAME path counts as mixed_rv_comparison.py, for a directly comparable
convergence table across all four approaches (plain, mixed-RV, antithetic,
CV).

ANTITHETIC VARIATES: draws n_paths//2 independent shock vectors Z and
mirrors each into -Z, so every path has an exact opposite-shock twin. For
any estimator that is (to first order) odd/linear in the driving normals,
averaging a Z-path with its -Z-twin cancels first-order sampling error --
typically tightens mean-based metrics (EE, EEPE) for the same total path
count, at zero extra simulation cost (still n_paths total paths, just
n_paths//2 independent draws).

CONTROL VARIATES, BOND FORWARDS ONLY: BondForwardTrade's NPV is
NPV = sign * (notional/face) * DF(T) * (forward_clean(T) - strike_clean),
and forward_clean(T) has an EXACT closed form (bond.forward_clean, a pure
function of the discount curve, forward date, and credit -- see
capitolis_pricers/bond.py). Under the calibrated LGM model, the true
(model, not simulated) expectation of the SIMULATED forward_clean(T) equals
the ANALYTIC forward_clean(T) computed off the t=0 calibration curve exactly
-- this is the LGM forward-matching guarantee this project has already
relied on and tested elsewhere (test_lgm_forward_matching.py). That gives a
control variate with a KNOWN true mean (the t=0 analytic forward), with no
extra simulation:

    NPV_cv = NPV_mc - beta * (forward_mc_avg - forward_analytic)

beta is estimated from the same simulated sample (beta_hat =
Cov(NPV_mc, forward_mc) / Var(forward_mc)) per path count.

Equity TRS and Bond TRS are NOT given a CV here: under GBM-SV, their payoff
has no simple closed form (stochastic volatility breaks the flat-vol
Black-Scholes-style analytic price), so no equally rigorous control variate
exists for them without introducing its own approximation error -- CV is
applied only where an exact analytic reference is available, per the
project's standing "no approximation without an explicit tradeoff
discussion" convention.

    python risk_engine/examples/variance_reduction_comparison.py
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
from risk_engine.pricing import price_curves, trade_maturity
from risk_engine.netting import build_netting_hierarchy
from risk_engine.exposure import compute_all_profiles
from risk_engine.simulation.scenario_market import build_market_states_at
from risk_engine.examples._sourced_book import build_sourced_book

PATH_COUNTS = [500, 1000, 2000, 5000, 10000, 20000]
REF_DATE = date(2026, 8, 24)


def _run_plain_or_antithetic(book, n_paths, antithetic, seed=42):
    ref_date, trades = book["ref_date"], book["trades"]
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    sim = book["build_simulator"]()
    rng = np.random.default_rng(seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                             rng=rng, ref_date=ref_date, antithetic=antithetic)
    result = price_curves(trades, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=None)
    counterparties = build_netting_hierarchy(trades)
    profiles = compute_all_profiles(counterparties, result, anchors, ref_date)
    return profiles["BOOK_TOTAL"], precache, result


_CV_CACHE = {}


def _bond_forward_cv_at_date(book, precache, trade, eval_date):
    """(forward_analytic, forward_mc_per_path) for ONE bond forward trade,
    evaluated at eval_date -- NOT necessarily the trade's own forward_date.

    This is the fix for the bug caught during smoke-testing: an earlier
    version of this function evaluated the control variate ONCE, at the
    trade's own settlement date, and then tried to use that single
    snapshot to correct exposure computed at a DIFFERENT date (the
    reporting anchor's MPoR window, d-1bd/d+10bd) -- covarying a control
    variate evaluated at one date against an NPV evaluated at another date
    is not a valid control-variate construction (the two are not measuring
    the same underlying randomness at the same point in time), and produced
    a nonsensical ~33% INCREASE in max_EE in testing rather than the
    variance reduction CV is supposed to provide.

    The correct construction evaluates the analytic AND simulated forward
    price at THE SAME eval_date as whatever NPV is being corrected --
    forward_clean(trade.forward_date) computed off a curve/state as of
    eval_date is still well-defined for eval_date < forward_date (it is a
    forward-starting quantity, not a spot payoff), matching exactly what
    the pricer itself evaluates when computing NPV at that scenario date.
    Results are cached per (trade id, eval_date, precache path count) since
    price_curves' own date loop already visits the same dates repeatedly
    (once for npv0 at d-1bd, once for npv10 at d+10bd across many reporting
    anchors -- many of which coincide across anchors for a short-dated
    book).

    CACHING NOTE (second real bug caught building this, this time in a
    downstream consumer -- single_trade_convergence.py): the cache key
    originally used id(precache) to distinguish different simulation runs.
    id() is a memory ADDRESS, not an identity guarantee across an object's
    full lifetime -- once an earlier path count's precache is garbage-
    collected (each loop iteration in single_trade_convergence.py discards
    the previous one), Python can and does reuse that same address for the
    NEXT precache object at a DIFFERENT path count. That produced a
    genuine crash: a 5,000-path run's forward_mc array (looked up via a
    stale cache hit keyed on a reused id()) came back with a 1,000-path
    shape, and the resulting size mismatch surfaced as a np.concatenate
    ValueError several calls downstream -- confirmed by reproducing the
    id()-reuse pattern directly. Fixed by keying on n_paths (an actual
    distinguishing value carried on precache's own state, not a transient
    memory address) instead of id(precache). A MODULE-LEVEL cache dict
    (not a mutable default argument) is used so it is shared correctly
    across calls without the mutable-default-argument trap either."""
    ref_date = book["ref_date"]
    usd_curve = book["market_data"]["usd_curve"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    # Keyed by n_paths AND a cheap content fingerprint (first rate factor's
    # first-path, first-date state value) rather than n_paths alone --
    # n_paths alone would silently conflate two DIFFERENT precache objects
    # at the same path count (e.g. a plain vs. antithetic run both at 5,000
    # paths) into one cache entry, which happens not to occur with this
    # module's current call sites (only ever called on the PLAIN run's
    # precache, one per n_paths) but is not guaranteed by the function
    # signature -- the fingerprint costs nothing extra to compute and
    # removes the assumption entirely.
    rate_state = next(iter(precache.rate_states.values()))
    n_paths = rate_state.shape[0]
    fingerprint = float(rate_state[0, 0, 0])
    key = (n_paths, fingerprint, trade.forward_date, eval_date)
    if key in _CV_CACHE:
        return _CV_CACHE[key]

    if eval_date <= ref_date:
        forward_analytic = trade.forward_clean_price(_AnalyticMarketStub(usd_curve, ref_date))
    else:
        # Analytic control at a FUTURE eval_date: the base-curve forward
        # price as seen from eval_date still has a closed form (curve
        # ratios shift consistently), computed via the same
        # _AnalyticMarketStub against the unchanged t=0 curve -- the
        # control variate's "true mean" is a property of the CALIBRATED
        # CURVE, not of any one scenario, so it does not need to roll
        # forward with eval_date; only the MC side (evaluated per path,
        # per eval_date, off the simulated state) changes with eval_date.
        forward_analytic = trade.forward_clean_price(_AnalyticMarketStub(usd_curve, ref_date))

    markets = build_market_states_at(precache, precache.sim_times, eval_date, equity_div)
    forward_mc_per_path = np.array([trade.forward_clean_price(m) for m in markets])

    _CV_CACHE[key] = (forward_analytic, forward_mc_per_path)
    return _CV_CACHE[key]


class _AnalyticMarketStub:
    """Just enough of a MarketState for BondForwardTrade._forward_clean's
    curve-derived path: market.discount(ccy) and market.credit(issuer)
    (None -- every bond in this book is risk-free per README Sec.7/8, so no
    issuer credit curve is ever populated in production MarketStates
    either)."""
    def __init__(self, usd_curve, ref_date):
        self._curve = usd_curve
        self.ref_date = ref_date

    def discount(self, ccy):
        return self._curve

    def credit(self, issuer):
        return None


def _cv_corrected_curve_result(book, precache, result, bond_forward_trades):
    """A NEW CurveResult with every bond-forward trade's npv0/npv10 entries
    CV-corrected at every regression date, everything else copied through
    unchanged -- then handed to the REAL exposure.compute_all_profiles, so
    every exclusion rule that function already implements (anchors with no
    valid prior VM mark dropped; trades maturing inside the MPoR window
    excluded from both curve legs) is reused exactly as-is rather than
    reimplemented.

    An earlier version of this comparison reimplemented the exposure/EE
    aggregation loop from scratch on top of raw npv0/npv10 dict lookups,
    to correct exposure per path before taking max(.,0). That reimplementation
    did NOT replicate either exclusion rule above, so its "plain" baseline
    silently disagreed with the real ExposureProfile's plain EE at the very
    same date (confirmed directly: date-1 plain EE reconstructed as $6.71M
    vs. the real engine's $2.77M for the identical inputs) -- the CV-vs-plain
    comparison was therefore invalid on its own terms, independent of
    whatever the CV correction itself was doing. Building a corrected
    CurveResult and handing it to the untouched, already-tested exposure
    engine avoids re-deriving (and re-risking) that logic a second time.
    """
    from dataclasses import replace
    from risk_engine.simulation.grid import add_business_days

    npv0 = dict(result.npv0)
    npv10 = dict(result.npv10)
    n_paths = result.n_paths
    mpor_days = result.mpor_days

    for tid, trade in bond_forward_trades.items():
        for d in result.regression_dates:
            # npv0[(p, d, tid)] is the value AT d -- control at d.
            if (0, d, tid) in npv0:
                fwd_analytic, fwd_mc = _bond_forward_cv_at_date(book, precache, trade, d)
                if fwd_mc.std() > 0:
                    npv_vals = np.array([npv0[(p, d, tid)] for p in range(n_paths)])
                    beta = np.cov(npv_vals, fwd_mc)[0, 1] / np.var(fwd_mc)
                    corrected = npv_vals - beta * (fwd_mc - fwd_analytic)
                    for p in range(n_paths):
                        npv0[(p, d, tid)] = float(corrected[p])

            # npv10[(p, d, tid)] is the value AT d + mpor_days (see
            # pricing.py's _price_path_chunk: markets10 is built at d10,
            # but the result dict is keyed by the ORIGINAL d, not d10) --
            # the control variate for THIS entry must therefore be
            # evaluated at d10, not d. Missing this distinction was the
            # second date-mismatch bug found while building this CV
            # implementation (the first was controlling npv0/npv10 with a
            # SINGLE forward-date evaluation instead of any per-leg date at
            # all; this one is using the anchor date d instead of d10 for
            # the npv10 leg specifically).
            if (0, d, tid) in npv10:
                d10 = add_business_days(d, mpor_days)
                fwd_analytic10, fwd_mc10 = _bond_forward_cv_at_date(book, precache, trade, d10)
                if fwd_mc10.std() > 0:
                    npv_vals10 = np.array([npv10[(p, d, tid)] for p in range(n_paths)])
                    beta10 = np.cov(npv_vals10, fwd_mc10)[0, 1] / np.var(fwd_mc10)
                    corrected10 = npv_vals10 - beta10 * (fwd_mc10 - fwd_analytic10)
                    for p in range(n_paths):
                        npv10[(p, d, tid)] = float(corrected10[p])

    return replace(result, npv0=npv0, npv10=npv10)


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    book = build_sourced_book(ref_date=REF_DATE)
    bond_forward_trades = {tid: t for tid, t in book["trades"].items() if isinstance(t, BondForwardTrade)}
    print(f"  {len(bond_forward_trades)} bond forward trade(s) eligible for CV: {list(bond_forward_trades)}")

    print(f"\n{'n_paths':>8s}  {'method':>11s}  {'max_EE':>14s}  {'MPE_99':>14s}  {'time(s)':>8s}")
    results = {}
    for n_paths in PATH_COUNTS:
        # -- plain --
        t0 = time.time()
        profile, precache, result = _run_plain_or_antithetic(book, n_paths, antithetic=False)
        elapsed = time.time() - t0
        max_ee = max(profile.ee) if profile.ee else 0.0
        results[(n_paths, "plain")] = (max_ee, profile.mpe_99)
        print(f"{n_paths:>8d}  {'plain':>11s}  {max_ee:>14,.2f}  {profile.mpe_99:>14,.2f}  {elapsed:>8.1f}")

        # -- antithetic --
        t0 = time.time()
        profile_a, _, _ = _run_plain_or_antithetic(book, n_paths, antithetic=True)
        elapsed = time.time() - t0
        max_ee_a = max(profile_a.ee) if profile_a.ee else 0.0
        results[(n_paths, "antithetic")] = (max_ee_a, profile_a.mpe_99)
        print(f"{n_paths:>8d}  {'antithetic':>11s}  {max_ee_a:>14,.2f}  {profile_a.mpe_99:>14,.2f}  {elapsed:>8.1f}")

        # -- control variate (reuses the PLAIN run's precache/result -- CV
        # adjusts the same sample, it does not need a fresh simulation).
        # Builds a CV-corrected CurveResult, then hands it to the REAL
        # exposure.compute_all_profiles (same function every other script
        # in this project uses) rather than reimplementing EE/PFE
        # aggregation -- see _cv_corrected_curve_result's docstring for why.
        t0 = time.time()
        cv_result = _cv_corrected_curve_result(book, precache, result, bond_forward_trades)
        counterparties = build_netting_hierarchy(book["trades"])
        cv_profiles = compute_all_profiles(counterparties, cv_result, book["anchors"], book["ref_date"])
        cv_profile = cv_profiles["BOOK_TOTAL"]
        elapsed = time.time() - t0
        max_ee_cv = max(cv_profile.ee) if cv_profile.ee else 0.0
        results[(n_paths, "cv")] = (max_ee_cv, cv_profile.mpe_99)
        print(f"{n_paths:>8d}  {'cv':>11s}  {max_ee_cv:>14,.2f}  {cv_profile.mpe_99:>14,.2f}  {elapsed:>8.1f}")

    print("\n=== Convergence: run-to-run stability in max_EE as n_paths increases ===")
    print(f"{'n_paths':>8s}  {'method':>11s}  {'max_EE':>14s}  {'|delta %|':>10s}")
    for method in ("plain", "antithetic", "cv"):
        prev = None
        for n_paths in PATH_COUNTS:
            max_ee, _ = results[(n_paths, method)]
            if prev is None:
                print(f"{n_paths:>8d}  {method:>11s}  {max_ee:>14,.2f}  {'--':>10s}")
            else:
                pct = abs(max_ee - prev) / prev * 100 if prev else float("nan")
                print(f"{n_paths:>8d}  {method:>11s}  {max_ee:>14,.2f}  {pct:>9.2f}%")
            prev = max_ee

    print("\n=== Convergence: run-to-run stability in MPE_99 as n_paths increases ===")
    print(f"{'n_paths':>8s}  {'method':>11s}  {'MPE_99':>14s}  {'|delta %|':>10s}")
    for method in ("plain", "antithetic", "cv"):
        prev = None
        for n_paths in PATH_COUNTS:
            _, mpe99 = results[(n_paths, method)]
            if prev is None:
                print(f"{n_paths:>8d}  {method:>11s}  {mpe99:>14,.2f}  {'--':>10s}")
            else:
                pct = abs(mpe99 - prev) / prev * 100 if prev else float("nan")
                print(f"{n_paths:>8d}  {method:>11s}  {mpe99:>14,.2f}  {pct:>9.2f}%")
            prev = mpe99


if __name__ == "__main__":
    main()
