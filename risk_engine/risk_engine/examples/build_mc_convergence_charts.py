"""
Generates real matplotlib charts from mc_convergence_analysis.py's actual
checkpoint results (benchmark_results/mc_convergence_checkpoint.json) --
20 real trials (4 path counts x 5 seeds), for build_mc_convergence_ppt.py
to embed. All numbers plotted here are the genuine measured trial results,
not fabricated or interpolated.

    python risk_engine/examples/build_mc_convergence_charts.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(HERE, "benchmark_results", "mc_convergence_checkpoint.json")

# ---------------------------------------------------------------- theme (matches build_methodology_ppt.py)
BG = "#05070F"
PANEL = "#0B1018"
CYAN = "#00E5FF"
MAGENTA = "#FF2EC4"
AMBER = "#FFB82E"
GREEN = "#39FF9D"
WHITE = "#F2F5FF"
GREY = "#8A93B8"
GRID = "#1C2444"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": WHITE, "text.color": WHITE,
    "xtick.color": GREY, "ytick.color": GREY, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.titlecolor": WHITE, "legend.facecolor": PANEL, "legend.edgecolor": GRID,
    "legend.labelcolor": WHITE,
})

PATH_COUNTS = [2000, 3000, 5000, 10000]
SEEDS = [1, 2, 3, 4, 5]
TRADE_IDS = ["BF_0003", "EQTRS_0001", "BTRS_0001"]
TRADE_COLORS = {"BF_0003": CYAN, "EQTRS_0001": MAGENTA, "BTRS_0001": AMBER}


def _save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def main():
    with open(CHECKPOINT_PATH) as fh:
        state = json.load(fh)
    analytic = state["analytic"]
    trials = state["trials"]

    def trial(n, s):
        return trials[f"{n}_{s}"]

    # ============================================================ Chart 1: NPV bias/RMSE per trade
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, tid in zip(axes, TRADE_IDS):
        means, rmses, biases_pct = [], [], []
        for n in PATH_COUNTS:
            vals = np.array([trial(n, s)["npvs"][tid] for s in SEEDS])
            means.append(vals.mean())
            errs = vals - analytic[tid]
            rmses.append(np.sqrt(np.mean(errs ** 2)))
            biases_pct.append((vals.mean() - analytic[tid]) / abs(analytic[tid]) * 100)
        ax.axhline(analytic[tid], color=WHITE, ls="--", lw=1.2, label="Analytic (exact)")
        ax.plot(PATH_COUNTS, means, "o-", color=TRADE_COLORS[tid], lw=2, ms=7, label="MC mean (5 seeds)")
        # scatter individual seeds
        for n in PATH_COUNTS:
            vals = [trial(n, s)["npvs"][tid] for s in SEEDS]
            ax.scatter([n] * len(vals), vals, color=TRADE_COLORS[tid], alpha=0.35, s=18, zorder=1)
        ax.set_title(f"{tid}\nbias @10k = {biases_pct[-1]:+.2f}%", fontsize=12)
        ax.set_xlabel("n_paths")
        ax.set_ylabel("NPV ($)")
        ax.ticklabel_format(style="plain", axis="y")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("MC NPV vs. Analytic Reference -- Mean (line), Individual Seeds (dots), Analytic (dashed)",
                 color=WHITE, fontsize=13)
    fig.tight_layout()
    _save(fig, "chart_mc_npv_vs_analytic.png")

    # ============================================================ Chart 2: RMSE% convergence (log-log)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tid in TRADE_IDS:
        rmse_pcts = []
        for n in PATH_COUNTS:
            vals = np.array([trial(n, s)["npvs"][tid] for s in SEEDS])
            errs = vals - analytic[tid]
            rmse = np.sqrt(np.mean(errs ** 2))
            rmse_pcts.append(rmse / abs(analytic[tid]) * 100)
        ax.plot(PATH_COUNTS, rmse_pcts, "o-", color=TRADE_COLORS[tid], lw=2, ms=8, label=tid)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_paths (log scale)")
    ax.set_ylabel("RMSE % of analytic NPV (log scale)")
    ax.set_title("Pricing RMSE% vs. Path Count -- BF_0003 converges;\nEQTRS_0001 / BTRS_0001 show BIAS-dominated (non-converging) error")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "chart_mc_rmse_convergence.png")

    # ============================================================ Chart 3: bias vs std decomposition
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(PATH_COUNTS))
    width = 0.35
    for ax, tid in zip(axes, TRADE_IDS):
        biases, stds = [], []
        for n in PATH_COUNTS:
            vals = np.array([trial(n, s)["npvs"][tid] for s in SEEDS])
            biases.append(abs(vals.mean() - analytic[tid]))
            stds.append(vals.std(ddof=1))
        ax.bar(x - width / 2, biases, width, color=MAGENTA, label="|Bias|")
        ax.bar(x + width / 2, stds, width, color=CYAN, label="Std (5 seeds)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in PATH_COUNTS])
        ax.set_xlabel("n_paths")
        ax.set_ylabel("$ magnitude")
        ax.set_title(tid, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Error Decomposition: |Bias| vs. Sampling Std -- bias >> std means MORE PATHS WON'T HELP",
                 color=WHITE, fontsize=13)
    fig.tight_layout()
    _save(fig, "chart_mc_bias_vs_std.png")

    # ============================================================ Chart 4: book-level exposure convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, key) in zip(axes, [("max_EE", "max_ee"), ("MPE_99", "mpe99")]):
        truth = np.mean([trial(10000, s)[key] for s in SEEDS])
        means, ses, rmse_pcts = [], [], []
        for n in PATH_COUNTS:
            vals = np.array([trial(n, s)[key] for s in SEEDS])
            means.append(vals.mean())
            ses.append(vals.std(ddof=1) / np.sqrt(len(vals)))
            rmse = np.sqrt(np.mean((vals - truth) ** 2))
            rmse_pcts.append(rmse / abs(truth) * 100)
        means = np.array(means)
        ses = np.array(ses)
        ax.axhline(truth, color=WHITE, ls="--", lw=1.2, label="Proxy truth (mean @10k)")
        ax.errorbar(PATH_COUNTS, means, yerr=1.96 * ses, fmt="o-", color=GREEN, lw=2, ms=8,
                    capsize=5, label="Mean +/- 95% CI (5 seeds)")
        ax2 = ax.twinx()
        ax2.plot(PATH_COUNTS, rmse_pcts, "s--", color=AMBER, lw=1.5, ms=6, alpha=0.8, label="RMSE % (right axis)")
        ax2.set_ylabel("RMSE % vs proxy truth", color=AMBER)
        ax2.tick_params(axis="y", colors=AMBER)
        ax.set_xlabel("n_paths")
        ax.set_ylabel(f"{label} ($)")
        ax.set_title(f"Book-level {label} convergence", fontsize=12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)
    fig.suptitle("Book-Level Exposure Convergence vs. Proxy Truth (mean of n=10,000 seeds)",
                 color=WHITE, fontsize=13)
    fig.tight_layout()
    _save(fig, "chart_mc_exposure_convergence.png")

    # ============================================================ Chart 5: runtime scaling
    fig, ax = plt.subplots(figsize=(8, 5.5))
    means_t, all_t = [], {}
    for n in PATH_COUNTS:
        times = [trial(n, s).get("elapsed_s", np.nan) for s in SEEDS]
        all_t[n] = times
        means_t.append(np.nanmean(times))
    for n in PATH_COUNTS:
        ax.scatter([n] * len(all_t[n]), all_t[n], color=GREY, alpha=0.5, s=30, zorder=1)
    ax.plot(PATH_COUNTS, means_t, "o-", color=CYAN, lw=2, ms=9, label="Mean runtime (5 seeds)", zorder=2)
    ax.set_xlabel("n_paths")
    ax.set_ylabel("Runtime per trial (s)")
    ax.set_title("Per-Trial Runtime vs. Path Count\n(n_workers=8; variance reflects shared-machine load, not algorithmic cost)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "chart_mc_runtime_scaling.png")

    print("\nAll MC convergence charts generated.")


if __name__ == "__main__":
    main()
