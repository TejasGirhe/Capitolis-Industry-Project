"""
EE / PFE_95 / PFE_99 exposure curve plots, one series set per counterparty
plus a book-wide total, MPE_99 marked as a horizontal reference line per
series. Renders headless (Agg backend) since this runs from a script/
notebook context, not an interactive session.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Dict

from .exposure import ExposureProfile


def plot_exposure_profiles(profiles: Dict[str, ExposureProfile], title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10.colors

    for i, (owner_id, profile) in enumerate(sorted(profiles.items())):
        color = colors[i % len(colors)]
        style = "--" if owner_id == "BOOK_TOTAL" else "-"
        lw = 2.5 if owner_id == "BOOK_TOTAL" else 1.6
        ax.plot(profile.dates, profile.ee, style, color=color, linewidth=lw, label=f"{owner_id} EE")
        ax.plot(profile.dates, profile.pfe_99, style, color=color, linewidth=lw, alpha=0.55, label=f"{owner_id} PFE (99%)")
        ax.plot(profile.dates, profile.pfe_95, style, color=color, linewidth=lw * 0.8, alpha=0.3, label=f"{owner_id} PFE (95%)")
        ax.axhline(profile.mpe_99, color=color, linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Horizon date")
    ax.set_ylabel("Exposure (USD)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
