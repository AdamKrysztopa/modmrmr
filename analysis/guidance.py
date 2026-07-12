"""Programmatic 'which criterion when' decision guide (contribution C2).

Ranks methods per task and per data regime (n>=p vs p>>n) from the tidy
results table, so the paper's decision guidance is derived, not hand-written.
"""

from __future__ import annotations

import pandas as pd

from analysis.ranks import auc_k_table, mean_ranks
from analysis.schema import data_regime


def rank_criteria(results: pd.DataFrame, task: str, datasets: list[str] | None = None) -> pd.Series:
    """Mean AUC-k rank per method for ``task`` (best-first), optionally on a subset."""
    sub = results[results["task"] == task]
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    return mean_ranks(auc_k_table(sub, task))


def build_decision_guide(results: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Return recommendations keyed by (task, data-regime).

    Each value is ``{"recommended": method, "ranking": [(method, mean_rank), ...]}``.
    """
    regime_of = {
        row.dataset: data_regime(int(row.n_samples), int(row.n_features))
        for row in results[["dataset", "n_samples", "n_features"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    guide: dict[tuple[str, str], dict] = {}
    for task in sorted(results["task"].unique()):
        task_datasets = results[results["task"] == task]["dataset"].unique()
        regimes = sorted({regime_of[d] for d in task_datasets})
        for regime in regimes:
            in_regime = [d for d in task_datasets if regime_of[d] == regime]
            ranks = rank_criteria(results, task, datasets=in_regime)
            guide[(task, regime)] = {
                "recommended": str(ranks.index[0]),
                "ranking": [(str(m), float(r)) for m, r in ranks.items()],
            }
    return guide
