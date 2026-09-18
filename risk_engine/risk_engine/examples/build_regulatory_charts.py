"""
Charts for the full regulatory-run status report, built from the real
full_regulatory_run_results.json output (10,000-path production run).

    python risk_engine/examples/build_regulatory_charts.py
"""
import json
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(HERE, "benchmark_results", "full_regulatory_run_results.json")

INK = "#2a2318"
PAPER = "#f7f5ef"
ACCENT = "#6b3a1f"
PASS = "#2f5233"
PARTIAL = "#9a6a1f"
SLATE = "#5b6470"
GRID = "#d8d3c4"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SLATE, "ytick.color": SLATE, "grid.color": GRID,
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12.5,
    "axes.titlecolor": INK, "legend.facecolor": PAPER, "legend.edgecolor": GRID,
})

CPTY_COLORS = {"CPTY_A": "#6b3a1f", "CPTY_B": "#9a6a1f", "CPTY_C": "#2f5233"}


def _save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def main():
    with open(RESULTS_PATH) as fh:
        d = json.load(fh)

    cptys = sorted(d["netting_set_level"].keys())

    # ---------------------------------------------------------- Chart 1: EE profiles, net vs gross
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    for ax, cpty in zip(axes, cptys):
        prof = d["netting_set_level"][cpty]["profile"]
        gross = d["netting_set_level"][cpty]["gross_ee"]
        dates = [date.fromisoformat(x) for x in prof["dates"]]
        ax.plot(dates, prof["ee"], "o-", color=CPTY_COLORS[cpty], lw=2, ms=4, label="Net EE (netting set)")
        if gross.get("gross_ee"):
            gdates = [date.fromisoformat(x) for x in gross["dates"]]
            ax.plot(gdates, gross["gross_ee"], "--", color=SLATE, lw=1.5, label="Gross EE (sum of trades)")
        ax.fill_between(dates, prof["pfe_95"], color=CPTY_COLORS[cpty], alpha=0.08)
        ax.plot(dates, prof["pfe_95"], ":", color=CPTY_COLORS[cpty], lw=1, alpha=0.7, label="PFE 95%")
        ax.plot(dates, prof["pfe_99"], ":", color=CPTY_COLORS[cpty], lw=1, alpha=0.4, label="PFE 99%")
        ax.set_title(cpty, fontsize=13)
        ax.set_xlabel("Reporting date")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Exposure (USD)")
    fig.suptitle("Expected Exposure: Net (netting-set) vs. Gross (sum of trades), with PFE bands",
                fontsize=14, color=INK)
    fig.tight_layout()
    _save(fig, "reg_ee_profiles.png")

    # ---------------------------------------------------------- Chart 2: netting benefit bar
    fig, ax = plt.subplots(figsize=(7, 5))
    net_vals = [d["netting_set_level"][c]["profile"]["mpe_95"] for c in cptys]
    gross_vals = [d["netting_set_level"][c]["gross_ee"].get("gross_max_ee", 0) for c in cptys]
    x = np.arange(len(cptys))
    width = 0.35
    ax.bar(x - width/2, gross_vals, width, color=SLATE, label="Gross max EE")
    ax.bar(x + width/2, net_vals, width, color=PASS, label="Net MPE (95%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cptys)
    ax.set_ylabel("USD")
    ax.set_title("Netting Impact by Counterparty")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "reg_netting_benefit.png")

    # ---------------------------------------------------------- Chart 3: xVA by counterparty
    fig, ax = plt.subplots(figsize=(8, 5.2))
    cva = [d["netting_set_level"][c]["xva"]["cva"] for c in cptys]
    dva = [d["netting_set_level"][c]["xva"]["dva"] for c in cptys]
    fva = [d["netting_set_level"][c]["xva"]["fva"] for c in cptys]
    x = np.arange(len(cptys))
    width = 0.25
    ax.bar(x - width, cva, width, color="#6b3a1f", label="CVA")
    ax.bar(x, dva, width, color="#2f5233", label="DVA")
    ax.bar(x + width, fva, width, color="#9a6a1f", label="FVA")
    ax.set_xticks(x)
    ax.set_xticklabels(cptys)
    ax.set_ylabel("USD")
    ax.set_title("xVA Components by Counterparty")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "reg_xva_by_cpty.png")

    # ---------------------------------------------------------- Chart 4: SA-CCR EAD waterfall-ish
    fig, ax = plt.subplots(figsize=(8, 5.2))
    rc = [d["netting_set_level"][c]["saccr"].get("RC", 0) for c in cptys]
    pfe = [d["netting_set_level"][c]["saccr"].get("PFE", 0) for c in cptys]
    ead = [d["netting_set_level"][c]["saccr"].get("EAD", 0) for c in cptys]
    x = np.arange(len(cptys))
    width = 0.25
    ax.bar(x - width, rc, width, color=SLATE, label="RC (Replacement Cost)")
    ax.bar(x, pfe, width, color=PARTIAL, label="PFE add-on")
    ax.bar(x + width, ead, width, color=ACCENT, label="EAD = α·(RC+PFE)")
    ax.set_xticks(x)
    ax.set_xticklabels(cptys)
    ax.set_ylabel("USD")
    ax.set_title("SA-CCR Exposure at Default (CRE52)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "reg_saccr_ead.png")

    print("\nAll regulatory-run charts generated.")


if __name__ == "__main__":
    main()
