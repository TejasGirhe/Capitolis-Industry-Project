"""
Report-specific charts for build_report_pdf.py, built purely from the saved
JSON outputs (no simulation):

  report_inputs.json      -- statics, MTM, SA-CCR, DV01
  exposure_profile.json   -- 10k-path EE / PFE / MPE term structures
  greeks_report.json      -- SA-CCR delta + vega (EE/MPE) bump-and-reprice
  xva_report.json         -- CVA / DVA / FVA
  xva_greeks.json         -- xVA vega

    python risk_engine/examples/report_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
OUT = HERE

BLUE = "#1e3c6e"
STEEL = "#4a7fb5"
GREEN = "#1e6e3c"
RED = "#963022"
GREY = "#8a8a8a"
CPTY_COLORS = {"CPTY_A": BLUE, "CPTY_B": STEEL, "CPTY_C": "#c98a2b"}


def _load(name):
    return json.load(open(os.path.join(HERE, name)))


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  {name}")
    return path


def _bar(ax, labels, values, colors=None, fmt="${:,.0f}", rotate=0):
    colors = colors or [BLUE] * len(labels)
    values = [float(v) for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=rotate)
    vmax = max(values) if values else 1.0
    vmin = min(values + [0.0])
    span = (vmax - vmin) or (abs(vmax) or 1.0)
    pad = 0.02 * span
    if vmax == vmin:
        ax.set_ylim(vmin - 1, vmax + 1)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + (pad if v >= 0 else -pad),
                fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.8)


def counterparty_panels(ri, exp, greeks, xva):
    cps = ["CPTY_A", "CPTY_B", "CPTY_C"]
    cols = [CPTY_COLORS[c] for c in cps]
    P = exp["profiles"]

    metrics = [
        ("Gross Notional", [ri["counterparties"][c]["gross_notional"] for c in cps], "${:,.0f}"),
        ("Trade Count", [ri["counterparties"][c]["trade_count"] for c in cps], "{:.0f}"),
        ("MTM at t0", [ri["counterparties"][c]["mtm0"] for c in cps], "${:,.0f}"),
        ("Max EE", [max(P[c]["ee"]) for c in cps], "${:,.0f}"),
        ("EEPE", [P[c]["eepe"] for c in cps], "${:,.0f}"),
        ("MPE 95%", [P[c]["mpe_95"] for c in cps], "${:,.0f}"),
        ("MPE 99%", [P[c]["mpe_99"] for c in cps], "${:,.0f}"),
        ("Net xVA", [xva["xva"][c]["net_xva"] for c in cps], "${:,.0f}"),
        ("IR DV01 (+1bp P&L)", [ri["counterparties"][c]["dv01"] for c in cps], "${:,.0f}"),
        ("SA-CCR Delta: Equity", [greeks["sa_ccr_delta"][c].get("equity", 0.0) for c in cps], "${:,.0f}"),
        ("Credit Spread (proxy)", [xva["counterparty_spreads"][c] * 1e4 for c in cps], "{:.0f}bp"),
        ("Notional-wtd TTM (yrs)", [ri["counterparties"][c]["notional_weighted_ttm"] for c in cps], "{:.2f}"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(13, 15))
    for ax, (title, vals, fmt) in zip(axes.flat, metrics):
        _bar(ax, cps, vals, cols, fmt=fmt)
        ax.set_title(title, fontsize=10, fontweight="bold", color=BLUE)
    fig.suptitle("Counterparty-Level Base Scenario Metrics (10,000 paths, ref 2026-08-24)",
                 fontsize=13, fontweight="bold", color=BLUE, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return _save(fig, "rpt_cpty_panels.png")


def saccr_panels(ri):
    cps = ["CPTY_A", "CPTY_B", "CPTY_C"]
    cols = [CPTY_COLORS[c] for c in cps]
    S = ri["saccr"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _bar(axes[0, 0], cps, [S[c]["RC"] for c in cps], cols); axes[0, 0].set_title("SA-CCR Replacement Cost (RC)", fontweight="bold", color=BLUE)
    _bar(axes[0, 1], cps, [S[c]["PFE"] for c in cps], cols); axes[0, 1].set_title("SA-CCR PFE (multiplier x aggregate add-on)", fontweight="bold", color=BLUE)
    _bar(axes[1, 0], cps, [S[c]["EAD"] for c in cps], cols); axes[1, 0].set_title("SA-CCR EAD = 1.4 x (RC + PFE)", fontweight="bold", color=BLUE)
    # add-on decomposition stacked
    ax = axes[1, 1]
    ir = [S[c]["addon_ir"] for c in cps]
    eq = [S[c]["addon_equity"] for c in cps]
    fx = [S[c]["addon_fx"] for c in cps]
    ax.bar(cps, ir, label="IR add-on", color=BLUE)
    ax.bar(cps, eq, bottom=ir, label="Equity add-on", color=STEEL)
    ax.bar(cps, fx, bottom=np.array(ir) + np.array(eq), label="FX add-on", color="#c98a2b")
    ax.set_title("SA-CCR Add-on Decomposition by Asset Class", fontweight="bold", color=BLUE)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for a in axes.flat:
        a.tick_params(axis="both", labelsize=8)
    fig.suptitle("SA-CCR Regulatory Exposure (uncollateralised netting sets, alpha = 1.4)",
                 fontsize=12.5, fontweight="bold", color=BLUE)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "rpt_saccr_panels.png")


def exposure_term_structure(exp):
    P = exp["profiles"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, c in zip(axes, ["CPTY_A", "CPTY_B", "CPTY_C"]):
        x = list(range(len(P[c]["dates"])))
        lab = [d[5:] for d in P[c]["dates"]]
        ax.plot(x, P[c]["ee"], marker="o", ms=3, label="EE", color=BLUE)
        ax.plot(x, P[c]["pfe_95"], marker="^", ms=3, label="PFE 95%", color=STEEL)
        ax.plot(x, P[c]["pfe_99"], marker="s", ms=3, label="PFE 99%", color=RED)
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=90, fontsize=6)
        ax.set_title(c, fontweight="bold", color=BLUE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Exposure (USD)")
    fig.suptitle("Exposure Term Structure by Counterparty (10k paths, MPoR = 10 business days)",
                 fontsize=12, fontweight="bold", color=BLUE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.22, wspace=0.2)
    return _save(fig, "rpt_exposure_term.png")


def vega_tornado(greeks):
    """Vega (Shocked - Base) in MPE 99% and EE_max, per factor group, book total."""
    runs = {r["factor_group"]: r["vega"]["BOOK_TOTAL"] for r in greeks["vega_runs"]}
    groups = ["rate", "equity", "fx"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ["EE_max", "MPE_99"],
                              ["Book dEE_max under +1bp vol shock", "Book dMPE99 under +1bp vol shock"]):
        vals = [runs[g][key] for g in groups]
        _bar(ax, [g.upper() for g in groups], vals,
             [BLUE, STEEL, "#c98a2b"], fmt="${:,.0f}")
        ax.set_title(title, fontweight="bold", color=BLUE, fontsize=10)
    fig.suptitle("Vega Bump-and-Reprice: Book-Total Exposure Sensitivity to +1bp Vol",
                 fontsize=11.5, fontweight="bold", color=BLUE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save(fig, "rpt_vega_tornado.png")


def xva_panels(xva, xva_vega):
    cps = ["CPTY_A", "CPTY_B", "CPTY_C"]
    cols = [CPTY_COLORS[c] for c in cps]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    X = xva["xva"]
    for ax, key, t in zip(axes, ["cva", "dva", "fva"], ["CVA", "DVA", "FVA"]):
        _bar(ax, cps, [X[c][key] for c in cps], cols, fmt="${:,.0f}")
        ax.set_title(t, fontweight="bold", color=BLUE)
    fig.suptitle("xVA by Counterparty (10k paths; rating-tier credit proxy, R = 40%)",
                 fontsize=12, fontweight="bold", color=BLUE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save(fig, "rpt_xva_panels.png")


def trade_level_bars(ri, dv01):
    tr = ri["trades"]
    tids = list(tr.keys())
    mtm = [tr[t]["mtm0"] for t in tids]
    notl = [tr[t]["notional"] for t in tids]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    cols = [CPTY_COLORS[tr[t]["counterparty"]] for t in tids]
    _bar(axes[0], tids, mtm, cols, fmt="${:,.0f}", rotate=90); axes[0].set_title("MTM at t0 by Trade", fontweight="bold", color=BLUE)
    _bar(axes[1], tids, notl, cols, fmt="${:,.0f}", rotate=90); axes[1].set_title("Notional by Trade", fontweight="bold", color=BLUE)
    ir_tids = [t for t in tids if t in dv01 and not isinstance(dv01[t], str)]
    _bar(axes[2], ir_tids, [dv01[t] for t in ir_tids],
         [CPTY_COLORS[tr[t]["counterparty"]] for t in ir_tids], fmt="${:,.0f}", rotate=90)
    axes[2].set_title("IR DV01 (+1bp parallel P&L) by Rate-Sensitive Trade", fontweight="bold", color=BLUE)
    fig.suptitle("Trade-Level Base Scenario (bar colour = counterparty)", fontsize=12, fontweight="bold", color=BLUE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "rpt_trade_bars.png")


def model_comparison():
    rows = {
        "LGM1F": (26692940.81, 19269475.60, 686611.20, 4731422.26),
        "LGM1F-SV": (26907146.26, 19289907.36, 685230.95, 4734059.53),
        "LGM2F": (25167376.28, 17954510.59, 627398.68, 4385498.07),
        "LGM2F-SV*": (27194888.78, 19474807.74, 651031.76, 4765362.10),
    }
    labels = list(rows)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for i, (ax, name) in enumerate(zip(axes, ["MPE 99%", "MPE 95%", "EEPE", "Max EE"])):
        vals = [rows[k][i] for k in labels]
        _bar(ax, labels, vals, [STEEL, STEEL, STEEL, BLUE], fmt="${:,.0f}", rotate=30)
        ax.set_title(name, fontweight="bold", color=BLUE, fontsize=9.5)
    fig.suptitle("Book-Level Exposure under Four LGM Rate-Model Specifications (10k paths, identical book)",
                 fontsize=11, fontweight="bold", color=BLUE)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "rpt_model_comparison.png")


def main():
    ri = _load("report_inputs.json")
    exp = _load("exposure_profile.json")
    greeks = _load("greeks_report.json")
    xva = _load("xva_report.json")
    xva_vega = _load("xva_greeks.json")
    print("Building report charts:")
    counterparty_panels(ri, exp, greeks, xva)
    saccr_panels(ri)
    exposure_term_structure(exp)
    vega_tornado(greeks)
    xva_panels(xva, xva_vega)
    trade_level_bars(ri, ri["dv01_by_trade"])
    model_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
