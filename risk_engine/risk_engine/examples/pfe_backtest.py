"""
Item #6 of the regulatory-readiness punch list (see
risk_engine/examples/regulatory_readiness_report.html): backtest simulated
PFE against SUBSEQUENTLY REALIZED market outcomes -- the one gap on that
list with no shortcut (SR 11-7 / model-validation-standard territory).

WHY THIS COULDN'T JUST REUSE build_sourced_book(ref_date=<past date>):
_sourced_book.py's market_data.equity.fetch_all_equity_history(range_="1y")
always fetches the trailing 1-year window from TODAY (Yahoo's chart API
`range` parameter is relative-to-now, not a fixed [start,end] query) --
passing a past ref_date does NOT truncate that history. Naively backtesting
against build_sourced_book(ref_date=<past>) would silently calibrate an
"as-of" book using REALIZED VOL AND SPOT DATA FROM AFTER THAT DATE -- a
textbook look-ahead-bias bug. Confirmed by reading market_data/equity.py
directly before writing this module.

This module fixes that by fetching each name's full available history
ONCE, then EXPLICITLY TRUNCATING to each as-of date's index before
computing spot/vol/correlation -- no data point after as_of_date is ever
used to build that window's "prediction."

Method, per as-of date T0 in AS_OF_DATES:
  1. Truncate every equity's price history to bars with date <= T0.
  2. From that truncated history alone: spot = last truncated close,
     realized vol = realized_vol() on truncated log-returns, correlation =
     realized_correlation_matrix() on truncated histories.
  3. Calibrate GBM_SV models off this "as-of-T0" market data, run a real
     Monte Carlo simulation of the SAME small equity basket forward to
     T0+HORIZON_DAYS, and read off the simulated PFE_95/PFE_99 quantile at
     that horizon (single-name, uncollateralized "exposure" = simulated
     price change; no rates/FX are stress-tested here, this is a first,
     equity-only PFE backtest, not the full book).
  4. Compare the REALIZED price at T0+HORIZON_DAYS (already in the fetched
     history, dated AFTER T0, never used to build step 2's calibration)
     against the predicted PFE_95/PFE_99 quantile: an "exceedance" is a
     realized outcome beyond the predicted quantile.
  5. Repeat across every date in AS_OF_DATES (spread across the available
     history) and every name in BACKTEST_TICKERS, then run a Kupiec
     proportion-of-failures test: at a 95% PFE quantile, ~5% of
     (date, name) observations SHOULD exceed it -- report the observed
     exceedance rate and whether it is statistically consistent with 5%/1%
     (chi-squared Kupiec LR test), not just "close enough" by eye.

SCOPE, stated explicitly rather than implied: this backtests SINGLE-NAME
EQUITY exposure only (GBM_SV, spot moves), not the full netted, multi-
currency, multi-trade-type book -- a deliberately smaller, tractable first
backtest that exercises the real methodology (as-of calibration -> forward
simulation -> compare to realized outcome -> Kupiec test) end-to-end,
rather than a book-level backtest that would need far more historical
windows than one year of daily data can support to be statistically
meaningful anyway.

    python risk_engine/examples/pfe_backtest.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.market_data.equity import fetch_equity_history
from risk_engine.market_data.vol_corr import realized_vol, realized_correlation_matrix
from risk_engine.calibration.market_surface import flat_vol_surface
from risk_engine.models.registry import get_spot_model, get_rate_model

# A small, liquid subset -- not the full 37-name book -- kept small so this
# script completes in a reasonable time and so per-(date,name) observations
# aren't so numerous that a handful of correlated bad days dominate the
# exceedance count (see the Kupiec test's independence caveat below).
BACKTEST_TICKERS = ["AAPL", "MSFT", "SPY"]
HORIZON_DAYS = 20            # ~1 trading month forward
N_PATHS = 5000
FLAT_RATE = 0.04             # placeholder short rate for THIS equity-only backtest's drift --
                              # not the book's real calibrated LGM curve, since this module
                              # tests exposure-quantile coverage, not full cross-asset pricing
CONFIDENCE_LEVELS = [0.95, 0.99]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "pfe_backtest_results.json")


def _dates_from_timestamps(timestamps):
    return [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in timestamps]


def _truncate_history(dates, closes, as_of_date):
    """Every (date, close) pair with date <= as_of_date -- the ONLY data
    this backtest's as-of-T0 calibration step is allowed to see."""
    idx = [i for i, d in enumerate(dates) if d <= as_of_date]
    return [dates[i] for i in idx], [closes[i] for i in idx]


def _realized_forward_return(dates, closes, as_of_date, horizon_days):
    """log(close at the first available trading date >= as_of_date +
    horizon_days) - log(close at as_of_date) -- the REALIZED outcome this
    backtest checks the model's predicted quantile against. Returns None if
    the fetched history doesn't extend far enough past as_of_date (can't
    score a window we don't have the "actual" for)."""
    target = as_of_date + timedelta(days=horizon_days)
    as_of_idx = max(i for i, d in enumerate(dates) if d <= as_of_date)
    future_idx = [i for i, d in enumerate(dates) if d >= target]
    if not future_idx:
        return None
    fut_i = future_idx[0]
    return float(np.log(closes[fut_i] / closes[as_of_idx]))


def _simulate_pfe_quantiles(spot, vol, rate, horizon_years, n_paths, confidence_levels, seed):
    """A single-name GBM_SV Monte Carlo of horizon-year forward log-return
    quantiles, calibrated off (spot, vol) alone -- this backtest's
    "exposure" is the simulated log-return distribution at the horizon,
    matched against the SAME log-return definition used in
    _realized_forward_return so predicted-vs-realized are apples-to-apples."""
    from datetime import date as date_cls
    ref = date_cls(2020, 1, 1)   # arbitrary anchor -- only relative horizon_years matters here
    horizon_date = ref + timedelta(days=int(horizon_years * 365))

    rate_model = get_rate_model("LGM1F").calibrate(
        __import__("capitolis_pricers.curves", fromlist=["zero_curve"]).zero_curve(
            ref, [0.1, 1.0, 5.0, 10.0], [rate, rate, rate, rate]),
        flat_vol_surface("RATE_USD", flat_vol=0.01))
    vol_surface = flat_vol_surface("EQ_BACKTEST", flat_vol=vol)
    spot_model = get_spot_model("GBM_SV").calibrate(spot, rate_model, vol_surface, dividend_rate=0.0)

    rng = np.random.default_rng(seed)
    rate_state = rate_model.simulate_paths(n_paths, [horizon_date], rng)
    rng2 = np.random.default_rng(seed + 1)
    spot_state = spot_model.simulate_paths(n_paths, [horizon_date], rng2)

    t = (horizon_date - ref).days / 365.0
    levels = np.array([
        spot_model.level_at(spot_state[p, 0, :], t, rate_state=rate_state[p, 0, :])
        for p in range(n_paths)
    ])
    log_returns = np.log(levels / spot)
    return {cl: float(np.quantile(log_returns, cl)) for cl in confidence_levels}


def kupiec_test(n_obs, n_exceptions, confidence_level):
    """Kupiec (1995) proportion-of-failures likelihood-ratio test: is the
    OBSERVED exceedance rate statistically consistent with the expected
    (1-confidence_level) rate, or does it reject at the 95% test-of-the-test
    level (chi-squared, 1 dof)? Returns (LR_statistic, p_value, rejects_95).
    """
    p_expected = 1.0 - confidence_level
    p_observed = n_exceptions / n_obs if n_obs > 0 else 0.0
    if n_exceptions == 0:
        # log(0) guard: LR still well-defined, just take the limit
        ln_l_null = n_obs * np.log(1 - p_expected)
        ln_l_alt = n_obs * np.log(1 - p_observed) if p_observed < 1 else 0.0
    elif n_exceptions == n_obs:
        ln_l_null = n_obs * np.log(p_expected)
        ln_l_alt = n_obs * np.log(p_observed)
    else:
        ln_l_null = (n_obs - n_exceptions) * np.log(1 - p_expected) + n_exceptions * np.log(p_expected)
        ln_l_alt = (n_obs - n_exceptions) * np.log(1 - p_observed) + n_exceptions * np.log(p_observed)
    lr_stat = -2.0 * (ln_l_null - ln_l_alt)
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_value, p_value < 0.05


def main():
    print(f"Fetching full available history for {BACKTEST_TICKERS} (Yahoo, up to 1y trailing from today)...")
    histories = {}
    for ticker in BACKTEST_TICKERS:
        h = fetch_equity_history(ticker, range_="1y", interval="1d")
        histories[ticker] = (_dates_from_timestamps(h["dates"]), h["close"])
        print(f"  {ticker}: {len(h['close'])} daily bars, {histories[ticker][0][0]} to {histories[ticker][0][-1]}")

    # As-of dates: every 10th trading day of the FIRST ticker's calendar,
    # skipping the tail HORIZON_DAYS so every window has a realized outcome
    # to score against, and skipping the first 60 bars so realized vol has
    # enough history to be meaningful.
    calendar = histories[BACKTEST_TICKERS[0]][0]
    candidate_idxs = range(60, len(calendar) - HORIZON_DAYS, 10)
    as_of_dates = [calendar[i] for i in candidate_idxs]
    print(f"\n{len(as_of_dates)} as-of dates x {len(BACKTEST_TICKERS)} names = "
          f"{len(as_of_dates) * len(BACKTEST_TICKERS)} backtest windows")

    horizon_years = HORIZON_DAYS / 365.0
    observations = []   # one dict per (ticker, as_of_date)

    for ticker in BACKTEST_TICKERS:
        dates, closes = histories[ticker]
        for as_of in as_of_dates:
            t_dates, t_closes = _truncate_history(dates, closes, as_of)
            if len(t_closes) < 30:
                continue
            realized_ret = _realized_forward_return(dates, closes, as_of, HORIZON_DAYS)
            if realized_ret is None:
                continue
            spot = t_closes[-1]
            vol = realized_vol(t_closes[-252:] if len(t_closes) > 252 else t_closes)
            seed = abs(hash((ticker, as_of.isoformat()))) % (2**31)
            quantiles = _simulate_pfe_quantiles(spot, vol, FLAT_RATE, horizon_years, N_PATHS,
                                                CONFIDENCE_LEVELS, seed)
            observations.append({
                "ticker": ticker, "as_of": as_of.isoformat(), "spot": spot, "realized_vol": vol,
                "realized_log_return": realized_ret, "predicted_quantiles": quantiles,
            })

    print(f"\n{len(observations)} scored (ticker, as_of) observations")

    print("\n" + "=" * 78)
    print("KUPIEC PROPORTION-OF-FAILURES TEST -- realized outcome vs. predicted PFE quantile")
    print("=" * 78)
    summary = {}
    for cl in CONFIDENCE_LEVELS:
        n_obs = len(observations)
        n_exceptions = sum(
            1 for o in observations
            if o["realized_log_return"] > o["predicted_quantiles"].get(cl, o["predicted_quantiles"].get(str(cl)))
        )
        exceedance_rate = n_exceptions / n_obs if n_obs else float("nan")
        expected_rate = 1.0 - cl
        lr_stat, p_value, rejects = kupiec_test(n_obs, n_exceptions, cl)
        summary[cl] = {"n_obs": n_obs, "n_exceptions": n_exceptions, "exceedance_rate": exceedance_rate,
                       "expected_rate": expected_rate, "kupiec_lr_stat": lr_stat, "kupiec_p_value": p_value,
                       "rejects_model_at_95pct": rejects}
        verdict = "REJECTS model calibration" if rejects else "consistent with model calibration"
        print(f"\nConfidence level {cl:.0%}  (expected exceedance rate = {expected_rate:.1%}):")
        print(f"  observed exceedances: {n_exceptions} / {n_obs}  ({exceedance_rate:.2%})")
        print(f"  Kupiec LR statistic: {lr_stat:.3f}   p-value: {p_value:.4f}")
        print(f"  Verdict: {verdict} (chi-squared test, alpha=0.05, 1 dof)")

    print("\nCAVEATS (stated explicitly, not omitted):")
    print("  - Single-name equity exposure only (AAPL/MSFT/SPY) -- not the full netted book,")
    print("    not multi-currency, no rates/FX stress. A first, tractable backtest that")
    print("    exercises the real as-of -> forward-simulate -> compare methodology.")
    print("  - As-of windows drawn from ONE overlapping year of daily history are NOT")
    print("    statistically independent (adjacent as-of dates share most of their forward")
    print("    window) -- the Kupiec test's classical i.i.d. assumption is only")
    print("    approximately met here, a known limitation of backtesting over one short,")
    print("    autocorrelated historical sample rather than many independent market regimes.")
    print("  - FLAT_RATE is a placeholder drift for this equity-only test, not the book's")
    print("    real calibrated curve.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "script": "pfe_backtest.py", "tickers": BACKTEST_TICKERS, "horizon_days": HORIZON_DAYS,
        "n_paths": N_PATHS, "confidence_levels": CONFIDENCE_LEVELS,
        "n_as_of_dates": len(as_of_dates), "n_observations": len(observations),
        "summary": {str(k): v for k, v in summary.items()},
        "observations": observations,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved reproducible results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
