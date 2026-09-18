"""
Generates the actual-vs-fitted calibration charts for the methodology PPT:

  1. USD / JPY zero-curve: real Bloomberg curve vs. the calibrated LGM2F_SV
     model's own discount-factor reconstruction, at the SAME pillar dates
     (should overlay near-exactly -- forward-matching by construction).
  2. USD / JPY ATM swaption vol smile-fit: real Bloomberg cube's ATM term
     structure vs. the model's own PiecewiseSigma-implied term vol (from
     bootstrap_sigma), at the longest tenor's smile (real cube points vs.
     the SV model's rho/eta-implied moneyness dependence).
  3. Equity/FX realized vs. model-implied vol term structure: the sourced
     realized-vol input vs. the calibrated GBM_SV/FXGBM_SV model's own
     PiecewiseSigma term structure (they are the SAME by construction for
     the term structure -- SV's THETA is a separate long-run level fit,
     shown too).
  4. Forward curve extension: today's fitted curve's own IMPLIED FORWARD
     rate, t to t+1Y, at each future date out to the book's horizon --
     shows the term structure the MODEL projects going forward, not just
     the spot-date curve.
  5. Forward-matching NUMERICAL PROOF: for the USD and JPY rate factors,
     simulate N Monte Carlo paths, and at each future grid date compute
     the AVERAGE simulated discount factor E[DF(0,t)] across paths, next
     to the ORIGINAL INPUT CURVE's own DF(0,t) -- these should agree
     within Monte Carlo noise, which is the actual, checkable evidence
     that LGM's forward-matching property holds in this implementation,
     not just a formula claim.

    python risk_engine/examples/build_calibration_charts.py
"""
import os
import sys
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.examples._sourced_book import build_sourced_book
from risk_engine.market_data import bloomberg_data as bd
from risk_engine.pricing import price_curves
from risk_engine.examples.analytic_vs_mc_convergence import _analytic_npv, _eval_date_for
from risk_engine.examples.variance_reduction_comparison import BondForwardTrade
from capitolis_pricers.pricers.equity_trs import EquityTRS
from capitolis_pricers.pricers.bond_trs import BondTRS

OUT_DIR = os.path.dirname(__file__)
REF_DATE = date(2026, 8, 31)

BLUE = "#1e3c6e"
STEEL = "#4a7fb5"
AMBER = "#b0781e"
RED = "#963022"
GREEN = "#1e6e3c"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color=BLUE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


def chart_curve_fit(curve, model, ref_date, label, out_path):
    """Real input curve vs. the calibrated LGM model's OWN discount-factor
    reconstruction, at a dense set of future dates -- these should overlay
    almost exactly (forward-matching by construction: the model's
    DF(t,T)=DF(0,T)/DF(0,t)*exp(...) formula is built to reproduce the
    input curve exactly given the deterministic t=0 state, zero
    accumulated variance)."""
    years = np.linspace(0.05, 30, 200)
    dates = [ref_date + timedelta(days=round(y * 365.0)) for y in years]
    curve_zero = [curve.zero_rate(d) * 100 for d in dates]

    # model reconstruction at t=0 (deterministic starting state -- zero
    # vector, zero accumulated variance -- IS the calibration curve itself
    # by the LGM formula's own construction; see discount_factor's t=0 case)
    n_factors = model.n_factors
    state0 = np.zeros(n_factors)
    model_zero = []
    for y, d in zip(years, dates):
        df = model.discount_factor(state0, 0.0, y, T_date=d, t_date=ref_date)
        model_zero.append(-np.log(df) / y * 100)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(years, curve_zero, color=BLUE, linewidth=2.5, label=f"{label} real curve (Bloomberg)")
    ax.plot(years, model_zero, color=AMBER, linewidth=1.5, linestyle="--", label=f"{label} LGM2F_SV reconstruction")
    _style(ax, f"{label}: Real Curve vs. Calibrated LGM Reconstruction", "Tenor (years)", "Zero rate (%, cont. comp.)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    max_abs_diff_bp = max(abs(c - m) for c, m in zip(curve_zero, model_zero)) * 100
    print(f"  {label} curve fit: max |real - model| = {max_abs_diff_bp:.4f} bp -> {out_path}")


def chart_swaption_smile(vol_surface, label, out_path, tenor_years=None):
    """Real Bloomberg ATM swaption cube's term structure (across expiries,
    at a fixed swap tenor) -- this project's cube is ATM-only (no genuine
    strike/moneyness axis), so the 'smile' shown here is the term
    structure across EXPIRY at a representative swap tenor, labeled
    honestly as such rather than implying a strike smile that doesn't
    exist in an ATM-only cube."""
    swap_tenor = tenor_years or vol_surface.strikes[len(vol_surface.strikes) // 2]
    expiries = sorted(vol_surface.tenors)
    vols = [vol_surface.vol(e, swap_tenor) * 10000 for e in expiries]   # decimal -> bp

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(expiries, vols, color=BLUE, marker="o", markersize=4, linewidth=2, label=f"{label} ATM normal vol (Bloomberg VCUB)")
    _style(ax, f"{label}: ATM Swaption Vol Term Structure ({swap_tenor:.0f}Y swap tenor)",
           "Option expiry (years)", "Normal vol (bp)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {label} swaption vol term structure -> {out_path}")


def chart_swaption_cube_slice(vol_surface, label, out_path):
    """Real cube's SWAP-TENOR axis at its LONGEST expiry -- this is the
    axis fit_skew_smile actually reads (>=3 points needed) to fit LGM-SV's
    rho/eta; shown here as the real data driving that fit, honestly
    labeled as a swap-tenor axis, not a strike/moneyness smile."""
    longest_expiry = max(vol_surface.tenors)
    swap_tenors = sorted(vol_surface.strikes)
    vols = [vol_surface.vol(longest_expiry, k) * 10000 for k in swap_tenors]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(swap_tenors, vols, color=AMBER, marker="s", markersize=4, linewidth=2,
           label=f"{label} ATM vol, {longest_expiry:.1f}Y expiry (Bloomberg VCUB)")
    _style(ax, f"{label}: Vol Across Swap Tenor -- Longest Expiry Slice\n(this axis feeds LGM-SV's fitted skew/curvature)",
           "Swap tenor (years)", "Normal vol (bp)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {label} swap-tenor slice (skew/curvature input) -> {out_path}")


def chart_equity_vol_term_structure(sourced_vol_surface, label, out_path):
    """Realized-vol input (the flat_vol_surface-shaped term structure this
    engine actually calibrates GBM_SV off) vs. itself post-bootstrap --
    for equity/FX, 'model term structure' IS the realized-vol input by
    construction (bootstrap_sigma reproduces the ATM term structure
    exactly, same forward-matching principle as rates) -- shown as the
    single sourced/fitted line since there is no separate second series to
    compare against (no equity/FX 'zero curve' concept)."""
    atm = sourced_vol_surface.atm_term_structure()
    tenors = [t for t, v in atm]
    vols = [v * 100 for t, v in atm]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(tenors, vols, color=STEEL, marker="o", markersize=5, linewidth=2, label=f"{label} realized-vol term structure (sourced input = model fit)")
    _style(ax, f"{label}: Realized Vol Term Structure (Model Input = Calibration Target)", "Tenor (years)", "Annualized vol (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {label} equity/FX vol term structure -> {out_path}")


def chart_forward_curve(curve, ref_date, label, out_path, horizon_years=10):
    """The curve's own IMPLIED FORWARD rate (t to t+1Y) at each future
    start date, vs. the SPOT zero-rate curve -- shows the term structure
    the model projects going forward from today, not just today's spot
    curve shape."""
    starts = np.linspace(0.05, horizon_years - 1, 150)
    fwd_rates, zero_rates = [], []
    for t0 in starts:
        t1 = t0 + 1.0
        d0 = ref_date + timedelta(days=round(t0 * 365.0))
        d1 = ref_date + timedelta(days=round(t1 * 365.0))
        df0, df1 = curve.discount(d0), curve.discount(d1)
        fwd = -np.log(df1 / df0) / (t1 - t0)
        fwd_rates.append(fwd * 100)
        zero_rates.append(curve.zero_rate(d0) * 100)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(starts, zero_rates, color=BLUE, linewidth=2, label=f"{label} spot zero rate")
    ax.plot(starts, fwd_rates, color=RED, linewidth=2, linestyle="--", label=f"{label} 1Y forward rate (t, t+1Y)")
    _style(ax, f"{label}: Spot Curve vs. Forward Curve (Today's Fit, Projected Forward)",
           "Start of period (years from ref date)", "Rate (%, cont. comp.)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {label} forward curve -> {out_path}")


def chart_forward_matching_proof(sim, market_state, grid, ref_date, rate_factor, curve, label, out_path, n_paths=5000):
    """THE numerical proof: simulate n_paths Monte Carlo paths for this
    rate factor, then at each grid date compute
        E_paths[ DF(0, t) ]  (average of the model's OWN discount_factor()
                              evaluated on each simulated path's state)
    and plot it against
        curve.discount(t)    (the ORIGINAL input curve's own DF(0,t))
    Agreement within Monte Carlo noise (visibly overlapping, small
    residual) is the actual checkable evidence that this implementation's
    forward-matching holds under real simulation, not just under the
    exact t=0 formula (chart_curve_fit's check) -- LGM's SDE construction
    guarantees E[DF(0,t)] = curve.discount(t) exactly in the continuous-
    time model; this confirms the discretized Monte Carlo simulation
    reproduces that expectation to within sampling error.
    """
    rng = np.random.default_rng(7)
    precache = sim.simulate(market_state, n_paths=n_paths, horizon_dates=grid.dates, rng=rng, ref_date=ref_date)
    model = precache.rate_models[rate_factor]
    cached = precache.rate_states[rate_factor]   # (n_paths, n_dates, n_state)

    sim_times = precache.sim_times
    mc_df, curve_df, dates_plot = [], [], []
    for d_idx, (t, d) in enumerate(zip(sim_times, grid.dates)):
        if t <= 0:
            continue
        dfs = np.array([model.discount_factor(cached[p, d_idx, :], 0.0, t) for p in range(n_paths)])
        mc_df.append(dfs.mean())
        curve_df.append(curve.discount(d))
        dates_plot.append(t)

    mc_df = np.array(mc_df)
    curve_df = np.array(curve_df)
    max_rel_diff_bp = float(np.max(np.abs(mc_df - curve_df) / curve_df)) * 10000

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dates_plot, curve_df, color=BLUE, linewidth=2.5, label=f"{label}: input curve DF(0,t)")
    ax.plot(dates_plot, mc_df, color=GREEN, linewidth=1.5, linestyle="--", marker="o", markersize=3,
           label=f"{label}: E[DF(0,t)] over {n_paths:,} MC paths")
    _style(ax, f"{label}: Forward-Matching Proof -- MC Average vs. Input Curve\n(max relative diff: {max_rel_diff_bp:.2f} bp)",
           "Time (years)", "Discount factor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {label} forward-matching MC proof: max rel diff = {max_rel_diff_bp:.2f} bp -> {out_path}")
    return max_rel_diff_bp


def chart_pricing_accuracy(book, sim, path_counts, trade_ids, out_path):
    """Analytic (t=0, curve-only, zero-vol -- exact, no MC noise) NPV vs.
    Monte Carlo mean NPV, across increasing path counts, for one
    representative trade of each product type (bond forward, equity TRS,
    bond TRS) -- reuses analytic_vs_mc_convergence.py's already-validated
    ground-truth construction directly rather than re-deriving it. Two
    panels: (a) |bias %| vs. path count per trade -- should shrink toward
    a small residual (or, for a seasoned trade with a known American-
    exercise/past-period limitation, plateau -- see that module's
    docstring for BTRS_0001's documented, non-bug plateau), and (b) the
    absolute analytic vs. one high-path-count MC NPV, per trade, as a bar
    comparison for an at-a-glance accuracy check."""
    trades = {tid: book["trades"][tid] for tid in trade_ids if tid in book["trades"]}
    eval_dates = {tid: _eval_date_for(t) for tid, t in trades.items()}
    analytic = {tid: _analytic_npv(t, book) for tid, t in trades.items()}

    grid, anchors, regression_dates = book["grid"], book["anchors"], book["regression_dates"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    bias_by_trade = {tid: [] for tid in trades}
    mc_at_max_paths = {}
    for n_paths in path_counts:
        rng = np.random.default_rng(42)
        precache = sim.simulate(book["market_state_for_correlation"], n_paths=n_paths, horizon_dates=grid.dates,
                                rng=rng, ref_date=book["ref_date"])
        result = price_curves(trades, precache, regression_dates, equity_dividend_rates=equity_div, n_workers=None)
        for tid in trades:
            npv_mc = result.mean_npv0_by_trade(eval_dates[tid])[tid]
            bias_pct = (npv_mc - analytic[tid]) / analytic[tid] * 100 if analytic[tid] else float("nan")
            bias_by_trade[tid].append(bias_pct)
            if n_paths == path_counts[-1]:
                mc_at_max_paths[tid] = npv_mc
        print(f"  pricing accuracy @ {n_paths} paths: " +
              ", ".join(f"{tid}={bias_by_trade[tid][-1]:+.3f}%" for tid in trades))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = [BLUE, AMBER, GREEN, STEEL, RED]
    for i, tid in enumerate(trades):
        ax1.plot(path_counts, [abs(b) for b in bias_by_trade[tid]], marker="o", color=colors[i % len(colors)],
                 label=f"{tid} ({type(trades[tid]).__name__})")
    ax1.set_xscale("log")
    _style(ax1, "Pricing Accuracy: |Bias| vs. Path Count\n(Monte Carlo mean vs. analytic curve-only NPV)",
           "Number of paths (log scale)", "|Bias| (%)")

    x = np.arange(len(trades))
    tids = list(trades.keys())
    analytic_vals = [analytic[t] for t in tids]
    mc_vals = [mc_at_max_paths[t] for t in tids]
    width = 0.35
    ax2.bar(x - width / 2, analytic_vals, width, color=BLUE, label="Analytic (exact)")
    ax2.bar(x + width / 2, mc_vals, width, color=AMBER, label=f"Monte Carlo ({path_counts[-1]:,} paths)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tids, rotation=15)
    _style(ax2, f"Analytic vs. Monte Carlo NPV\n(at {path_counts[-1]:,} paths)", "Trade", "NPV (USD)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  pricing accuracy charts -> {out_path}")
    return bias_by_trade, analytic, mc_at_max_paths


def main():
    print(f"Loading book (ref date {REF_DATE})...")
    book = build_sourced_book(ref_date=REF_DATE)
    usd_curve = book["market_data"]["usd_curve"]
    rate_calibrated = book["rate_calibrated"]

    sim = book["build_simulator"]()
    jpy_curve_vol = [m for f, m in sim._rate_factors if str(f) == "RATE_JPY"]
    jpy_rate_calibrated = jpy_curve_vol[0] if jpy_curve_vol else None

    print("\n--- Rate curve fits ---")
    chart_curve_fit(usd_curve, rate_calibrated, REF_DATE, "USD SOFR", os.path.join(OUT_DIR, "chart_usd_curve_fit.png"))
    chart_forward_curve(usd_curve, REF_DATE, "USD SOFR", os.path.join(OUT_DIR, "chart_usd_forward_curve.png"))

    jpy_curve = bd.load_jpy_curve(REF_DATE)
    if jpy_rate_calibrated is not None:
        chart_curve_fit(jpy_curve, jpy_rate_calibrated, REF_DATE, "JPY OIS", os.path.join(OUT_DIR, "chart_jpy_curve_fit.png"))
    chart_forward_curve(jpy_curve, REF_DATE, "JPY OIS", os.path.join(OUT_DIR, "chart_jpy_forward_curve.png"))

    print("\n--- Swaption vol fits ---")
    usd_swap_vol = bd.load_usd_swaption_vol_surface()
    chart_swaption_smile(usd_swap_vol, "USD SOFR", os.path.join(OUT_DIR, "chart_usd_swaption_term.png"))
    chart_swaption_cube_slice(usd_swap_vol, "USD SOFR", os.path.join(OUT_DIR, "chart_usd_swaption_slice.png"))

    jpy_swap_vol = bd.load_jpy_swaption_vol_surface()
    chart_swaption_smile(jpy_swap_vol, "JPY OIS", os.path.join(OUT_DIR, "chart_jpy_swaption_term.png"))
    chart_swaption_cube_slice(jpy_swap_vol, "JPY OIS", os.path.join(OUT_DIR, "chart_jpy_swaption_slice.png"))

    print("\n--- Equity/FX vol term structures ---")
    # one representative US-listed and one JPY-listed equity name
    eq_factors = book["factors"].equities
    us_eq = next((e for e in eq_factors if e.native_ccy == "USD" and e.isin in book["market_data"]["equity_vol_surface"]), None)
    jp_eq = next((e for e in eq_factors if e.native_ccy == "JPY" and e.isin in book["market_data"]["equity_vol_surface"]), None)
    if us_eq:
        chart_equity_vol_term_structure(book["market_data"]["equity_vol_surface"][us_eq.isin],
                                        f"Equity {us_eq.isin} (USD)", os.path.join(OUT_DIR, "chart_equity_usd_vol.png"))
    if jp_eq:
        chart_equity_vol_term_structure(book["market_data"]["equity_vol_surface"][jp_eq.isin],
                                        f"Equity {jp_eq.isin} (JPY)", os.path.join(OUT_DIR, "chart_equity_jpy_vol.png"))
    if book["market_data"].get("fx_vol_surface") is not None:
        chart_equity_vol_term_structure(book["market_data"]["fx_vol_surface"], "USD/JPY FX",
                                        os.path.join(OUT_DIR, "chart_fx_vol.png"))

    print("\n--- Forward-matching numerical proof (Monte Carlo) ---")
    usd_factor = book["factors"].rates[0]
    usd_diff_bp = chart_forward_matching_proof(sim, book["market_state_for_correlation"], book["grid"], REF_DATE,
                                                usd_factor, usd_curve, "USD SOFR",
                                                os.path.join(OUT_DIR, "chart_usd_forward_matching_proof.png"))
    if jpy_rate_calibrated is not None:
        from risk_engine.factors.types import RateFactor
        jpy_factor = RateFactor(currency="JPY")
        jpy_diff_bp = chart_forward_matching_proof(sim, book["market_state_for_correlation"], book["grid"], REF_DATE,
                                                    jpy_factor, jpy_curve, "JPY OIS",
                                                    os.path.join(OUT_DIR, "chart_jpy_forward_matching_proof.png"))

    print("\n--- Pricing accuracy: analytic vs Monte Carlo ---")
    accuracy_trade_ids = [tid for tid in ("BF_0003", "EQTRS_0001", "BTRS_0001") if tid in book["trades"]]
    if accuracy_trade_ids:
        chart_pricing_accuracy(book, sim, [500, 1000, 2000, 5000, 10000], accuracy_trade_ids,
                               os.path.join(OUT_DIR, "chart_pricing_accuracy.png"))

    print("\nAll charts saved to", OUT_DIR)


if __name__ == "__main__":
    main()
