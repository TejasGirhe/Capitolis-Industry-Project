"""
Per-counterparty cashflow/reset/maturity date graph -- a structural view of
the book (the dates trades themselves define), not a simulated quantity.
Stacked bar by date, one bar segment per trade contributing a cashflow date.
"""
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..simulation.grid import trade_cashflow_dates


def plot_counterparty_cashflow_graph(counterparty, trades: dict, out_path: str, title: str = None):
    trade_ids = [tid for ns in counterparty.netting_sets for tid in ns.trade_ids]
    date_counts = defaultdict(lambda: defaultdict(int))
    for tid in trade_ids:
        trade = trades[tid]
        for d in trade_cashflow_dates(trade):
            date_counts[d][tid] += 1

    dates = sorted(date_counts.keys())
    if not dates:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    bottoms = [0] * len(dates)
    seen_trades = sorted({tid for counts in date_counts.values() for tid in counts})
    colors = plt.cm.tab20.colors
    for i, tid in enumerate(seen_trades):
        heights = [date_counts[d].get(tid, 0) for d in dates]
        ax.bar(dates, heights, bottom=bottoms, width=3, label=tid, color=colors[i % len(colors)])
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_title(title or f"{counterparty.id} -- cashflow/reset/maturity dates")
    ax.set_ylabel("Cashflow events")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all_cashflow_graphs(counterparties, trades: dict, out_dir: str) -> dict:
    paths = {}
    for cpty in counterparties:
        out_path = os.path.join(out_dir, f"cashflow_{cpty.id}.png")
        result = plot_counterparty_cashflow_graph(cpty, trades, out_path)
        if result:
            paths[cpty.id] = result
    return paths
