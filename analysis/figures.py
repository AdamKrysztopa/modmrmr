"""Paper figure generators. Each takes the tidy results table and writes one
figure file, returning its path. Headless (Agg) so it runs under CI/pytest.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
from scipy.stats import friedmanchisquare  # noqa: E402

from analysis.ranks import auc_k_table, mean_ranks  # noqa: E402
from analysis.schema import PRIMARY_METRIC  # noqa: E402

_OPERATORS = ["difference", "quotient", "multiplicative"]
_AGGREGATIONS = ["mean", "max", "sum"]


def cd_diagram(results: pd.DataFrame, task: str, out_path: str | Path) -> Path:
    """Friedman + Nemenyi critical-difference diagram over AUC-k ranks."""
    out_path = Path(out_path)
    scores = auc_k_table(results, task)
    # Friedman test across datasets (informational; drawn regardless).
    friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    avg_ranks = scores.rank(axis=1, ascending=False).mean(axis=0)
    sig_matrix = sp.posthoc_nemenyi_friedman(scores)
    plt.figure(figsize=(9, 2.6))
    sp.critical_difference_diagram(avg_ranks, sig_matrix)
    plt.title(f"Critical-difference diagram — {task} (AUC over k)")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    return out_path


def grid_heatmap(results: pd.DataFrame, task: str, out_path: str | Path) -> Path:
    """Heatmap of mean rank per operator x aggregation cell (in-family only)."""
    out_path = Path(out_path)
    ranks = mean_ranks(auc_k_table(results, task))
    # Map each in-family method to its (operator, aggregation) cell.
    meta = (
        results[results["task"] == task][["method", "operator", "aggregation"]]
        .drop_duplicates()
        .set_index("method")
    )
    grid = pd.DataFrame(index=_AGGREGATIONS, columns=_OPERATORS, dtype=float)
    for method, rank in ranks.items():
        op = meta.loc[method, "operator"]
        agg = meta.loc[method, "aggregation"]
        if op in _OPERATORS and agg in _AGGREGATIONS:
            grid.loc[agg, op] = float(rank)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    data = grid.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(_OPERATORS)), _OPERATORS, rotation=20, ha="right")
    ax.set_yticks(range(len(_AGGREGATIONS)), _AGGREGATIONS)
    ax.set_xlabel("operator")
    ax.set_ylabel("aggregation")
    ax.set_title(f"Mean rank by cell — {task}")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax, label="mean rank (lower is better)")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def auc_k_barchart(results: pd.DataFrame, task: str, out_path: str | Path) -> Path:
    """Bar chart of mean AUC-over-k per method (error bars = std over datasets)."""
    out_path = Path(out_path)
    scores = auc_k_table(results, task)
    means = scores.mean(axis=0).sort_values(ascending=False)
    stds = scores.std(axis=0).reindex(means.index)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(len(means)), means.to_numpy(), yerr=stds.to_numpy(), capsize=3)
    ax.set_xticks(range(len(means)), list(means.index), rotation=25, ha="right")
    ax.set_ylabel(f"mean AUC-k ({PRIMARY_METRIC[task]})")
    ax.set_title(f"Area under the k-curve — {task}")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def stability_accuracy_scatter(results: pd.DataFrame, task: str, out_path: str | Path) -> Path:
    """Scatter of mean stability (x) vs mean AUC-k (y), one point per method."""
    out_path = Path(out_path)
    scores = auc_k_table(results, task)
    acc = scores.mean(axis=0)
    stab = results[results["task"] == task].groupby("method")["stability"].mean().reindex(acc.index)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(stab.to_numpy(), acc.to_numpy())
    for method in acc.index:
        ax.annotate(
            method,
            (stab[method], acc[method]),
            fontsize=8,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("mean stability (Nogueira 2018)")
    ax.set_ylabel(f"mean AUC-k ({PRIMARY_METRIC[task]})")
    ax.set_title(f"Stability vs accuracy — {task}")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path
