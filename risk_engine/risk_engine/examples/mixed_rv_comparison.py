"""
EXPERIMENTAL comparison: pure-Gaussian driving shocks (the engine's actual,
production behavior) versus a "mixed" shock built as a standardized sum of
three differently-distributed random variables -- Normal + Student-t(df=5)
+ Uniform(-1,1), each drawn independently and summed, then standardized to
mean 0 / variance 1 before entering the SAME Cholesky correlation step
every other simulation in this engine uses (JointSimulator.simulate's
shock_sampler hook, added specifically for this comparison -- see its
docstring in simulation/joint.py).

WHY THIS IS EXPERIMENTAL, NOT A PRODUCTION OPTION:
Every closed-form formula in this engine -- LGM's discount-factor
reconstruction (DF(t,T) = ... exp(-Hx - 0.5 H^2 zeta)), GBM/FXGBM's level
reconstruction, and the Brownian-bridge interpolation used at every off-grid
regression date -- is DERIVED assuming the driving increments are Gaussian.
A mixed/summed-RV shock is close to Gaussian by the Central Limit Theorem
(summing only 3 components does not get all the way there -- some excess
kurtosis survives, especially from the Student-t(df=5) tail), so these
formulas become approximately rather than exactly correct under a mixed
shock. This script exists to quantify the resulting path-count convergence
behavior side by side with pure Gaussian, not to recommend switching
production to it.

For each path count in PATH_COUNTS, this script:
  1. runs the production book once with pure-Gaussian shocks,
  2. runs it again with the mixed shock (same seed, same everything else),
  3. reports EE/MPE_99 and each result's Monte Carlo standard error
     (std across paths / sqrt(n_paths), the standard MC convergence
     diagnostic already used elsewhere in this project).

    python risk_engine/examples/mixed_rv_comparison.py
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

PATH_COUNTS = [500, 1000, 2000, 5000, 10000, 20000]
REF_DATE = date(2026, 8, 24)


def mixed_shock_sampler(rng, shape):
    """Normal(0,1) + Student-t(df=5) + Uniform(-1,1), summed elementwise,
    then standardized to mean 0 / variance 1. Each component drawn
    independently at every (path, step, driver) cell. Uses scipy's t/uniform
    RVS with this call's rng as the numpy Generator they accept directly."""
    z_normal = rng.standard_normal(shape)
    z_t = stats.t.rvs(df=5, size=shape, random_state=rng)
    z_uniform = rng.uniform(-1.0, 1.0, size=shape)
    raw = z_normal + z_t + z_uniform
    return (raw - raw.mean()) / raw.std()


def _run(book, n_paths, shock_sampler, seed=42):
    ref_date, trades = book["ref_date"], book["trades"]
    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    sim = book["build_simulator"]()
    rng = np.random.default_rng(seed)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                             rng=rng, ref_date=ref_date, shock_sampler=shock_sampler)
    result = price_curves(trades, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=None)
    counterparties = build_netting_hierarchy(trades)
    profiles = compute_all_profiles(counterparties, result, anchors, ref_date)
    return profiles["BOOK_TOTAL"]


def _mc_stderr_of_ee(profile, n_paths):
    """Approximate MC standard error of the EE curve's PEAK value: EE is
    itself a per-date MEAN across paths, so its own standard error is
    std(per-path exposure at that date)/sqrt(n_paths) -- but ExposureProfile
    only stores the already-averaged EE, not per-path exposures, so this
    reports the standard error of the EE CURVE across reporting dates as a
    convergence proxy instead (how much the peak-EE estimate itself would be
    expected to jitter run-to-run at this path count, approximated from the
    spread of EE values already computed). A precise per-date MC stderr
    would require per-path exposures, not exposed by ExposureProfile today.
    """
    ee = np.array(profile.ee)
    return ee.std(ddof=1) / np.sqrt(n_paths) if len(ee) > 1 else float("nan")


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    book = build_sourced_book(ref_date=REF_DATE)

    print(f"\n{'n_paths':>8s}  {'variant':>8s}  {'max_EE':>14s}  {'MPE_99':>14s}  {'EEPE':>12s}  {'time(s)':>8s}")
    results = {}
    for n_paths in PATH_COUNTS:
        for variant, sampler in [("gaussian", None), ("mixed", mixed_shock_sampler)]:
            t0 = time.time()
            profile = _run(book, n_paths, sampler)
            elapsed = time.time() - t0
            max_ee = max(profile.ee) if profile.ee else 0.0
            results[(n_paths, variant)] = profile
            print(f"{n_paths:>8d}  {variant:>8s}  {max_ee:>14,.2f}  {profile.mpe_99:>14,.2f}  "
                  f"{profile.eepe:>12,.2f}  {elapsed:>8.1f}")

    print("\n=== Convergence: run-to-run stability as n_paths increases ===")
    print(f"{'n_paths':>8s}  {'variant':>8s}  {'max_EE':>14s}  {'|delta from prior N|':>20s}  {'|delta %|':>10s}")
    for variant in ("gaussian", "mixed"):
        prev_ee = None
        for n_paths in PATH_COUNTS:
            profile = results[(n_paths, variant)]
            max_ee = max(profile.ee) if profile.ee else 0.0
            if prev_ee is None:
                print(f"{n_paths:>8d}  {variant:>8s}  {max_ee:>14,.2f}  {'--':>20s}  {'--':>10s}")
            else:
                delta = max_ee - prev_ee
                pct = abs(delta) / prev_ee * 100 if prev_ee else float("nan")
                print(f"{n_paths:>8d}  {variant:>8s}  {max_ee:>14,.2f}  {delta:>20,.2f}  {pct:>9.2f}%")
            prev_ee = max_ee

    print("\nMinimum path count guidance: look for the smallest n_paths above which the "
          "run-to-run |delta %| in max_EE stays below a few percent for BOTH variants -- "
          "that is the practical convergence floor for this book at this reporting grid.")


if __name__ == "__main__":
    main()
