"""Cross-dataset statistics: Friedman + Nemenyi + critical-difference diagram.

Follows Demsar 2006: rank methods per dataset, Friedman test on mean ranks, Nemenyi
post-hoc, and a critical-difference (CD) diagram. Reported separately per task by the
caller (classification vs regression). Consumes a tidy results frame like
``run_grid``'s output (needs at least ``method, dataset, task, score``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.stats import friedmanchisquare  # noqa: E402


def _pivot(results: pd.DataFrame, task: str) -> pd.DataFrame:
    """Datasets (rows) x methods (cols) matrix of mean score for one task."""
    subset = results[results["task"] == task]
    if subset.empty:
        raise ValueError(f"No rows for task={task!r}")
    return subset.pivot_table(index="dataset", columns="method", values="score", aggfunc="mean")


def friedman_pvalue(results: pd.DataFrame, task: str) -> float:
    """Friedman test p-value across datasets (each method is a repeated measure)."""
    pivot = _pivot(results, task).dropna(axis=0, how="any")
    if pivot.shape[1] < 3:
        raise ValueError("Friedman test needs >= 3 methods")
    samples = [pivot[col].to_numpy() for col in pivot.columns]
    _, p = friedmanchisquare(*samples)
    return float(p)


def average_ranks(results: pd.DataFrame, task: str) -> pd.Series:
    """Mean rank per method across datasets; rank 1 = best (higher score is better)."""
    pivot = _pivot(results, task).dropna(axis=0, how="any")
    # rank within each dataset row; negate so the highest score gets rank 1.
    ranks = (-pivot).rank(axis=1, method="average")
    return ranks.mean(axis=0).sort_values()


def critical_difference(results: pd.DataFrame, task: str) -> Figure:
    """Render a Friedman + Nemenyi critical-difference diagram for one task."""
    pivot = _pivot(results, task).dropna(axis=0, how="any")
    ranks = average_ranks(results, task)
    # Nemenyi post-hoc p-values on the datasets x methods matrix.
    nemenyi = sp.posthoc_nemenyi_friedman(pivot.to_numpy())
    nemenyi.index = pivot.columns
    nemenyi.columns = pivot.columns

    fig, ax = plt.subplots(figsize=(8, 2.5))
    sp.critical_difference_diagram(
        ranks=ranks,
        sig_matrix=nemenyi,
        ax=ax,
        label_fmt_left="{label} ({rank:.2f})  ",
        label_fmt_right="  ({rank:.2f}) {label}",
    )
    ax.set_title(f"Critical-difference diagram ({task})")
    fig.tight_layout()
    return fig
