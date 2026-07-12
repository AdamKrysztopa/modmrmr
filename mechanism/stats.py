"""Friedman + Nemenyi significance over datasets (Demšar 2006) for the paper's claims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class FriedmanNemenyiResult:
    """Result of a Friedman test + Nemenyi post-hoc pairwise comparison.

    ``avg_ranks`` and ``nemenyi_p`` are always oriented so that a lower
    average rank means a better treatment (i.e. the ``higher_is_better``
    flag passed to :func:`friedman_nemenyi` has already been applied).
    """

    statistic: float
    p_value: float
    avg_ranks: pd.Series  # index=treatment; lower = better
    nemenyi_p: pd.DataFrame  # pairwise Nemenyi p-values, treatments x treatments
    n_blocks: int


def friedman_nemenyi(
    df: pd.DataFrame,
    *,
    treatment: str,
    block: str,
    value: str,
    higher_is_better: bool = True,
) -> FriedmanNemenyiResult:
    """Friedman test + Nemenyi post-hoc pairwise significance over ``block`` (datasets).

    ``df`` is a long-format frame with one row per (block, treatment) observation.
    Rows with missing treatments for a given block are dropped (Friedman requires a
    complete block design).
    """
    import scikit_posthocs as sp  # benchmarks dependency group

    pivot = df.pivot_table(index=block, columns=treatment, values=value, aggfunc="mean").dropna()
    if pivot.shape[1] < 3:
        raise ValueError("friedman_nemenyi requires at least 3 treatments")

    statistic, p_value = scipy_stats.friedmanchisquare(*[pivot[c] for c in pivot.columns])

    # Rank-transform explicitly so "lower rank = better" holds under both orientations,
    # and feed the *ranks themselves* into the Nemenyi post-hoc so its internal
    # ranking (which the library recomputes from raw values) matches ours exactly
    # regardless of higher_is_better.
    ranks = pivot.rank(axis=1, ascending=not higher_is_better)
    avg_ranks = ranks.mean(axis=0)

    nemenyi_p = sp.posthoc_nemenyi_friedman(ranks.reset_index(drop=True))
    nemenyi_p.index = pivot.columns
    nemenyi_p.columns = pivot.columns

    return FriedmanNemenyiResult(
        statistic=float(statistic),
        p_value=float(p_value),
        avg_ranks=avg_ranks,
        nemenyi_p=nemenyi_p,
        n_blocks=int(pivot.shape[0]),
    )


def plot_critical_difference(
    result: FriedmanNemenyiResult, out_path: Path | str, *, title: str = ""
) -> Path:
    """Render a critical-difference diagram (Demšar 2006) for ``result`` to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    fig, ax = plt.subplots(figsize=(7, 2.2))
    sp.critical_difference_diagram(result.avg_ranks, result.nemenyi_p, ax=ax)
    if title:
        ax.set_title(title)
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
