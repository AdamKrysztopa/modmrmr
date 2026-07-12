"""Shared ranking backbone consumed by figures, tables, and guidance.

Everything ranks on the per-task primary metric (higher-is-better), summarized
per (dataset, method) as the area under the feature-count (k) curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.schema import PRIMARY_METRIC


def primary_metric(task: str) -> str:
    """Return the higher-is-better ranking metric for ``task``."""
    return PRIMARY_METRIC[task]


def auc_k_table(results: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return a dataset x method matrix of area-under-the-k-curve scores.

    For each (dataset, method): average the primary metric over fold first (per
    dataset x method x k x learner x seed, per the schema), then over learner and
    seed at each k, integrate over k with the trapezoidal rule, and normalize by
    the k-range so the value reads as a mean primary score. Aggregating fold first
    keeps each (learner, seed) weighted equally even if fold counts are uneven.
    """
    metric = primary_metric(task)
    df = results[(results["task"] == task) & (results["metric"] == metric)]
    per_fold = df.groupby(["dataset", "method", "k", "learner", "seed"], as_index=False)[
        "score"
    ].mean()
    per_k = per_fold.groupby(["dataset", "method", "k"], as_index=False)["score"].mean()
    records: list[dict[str, object]] = []
    for (dataset, method), g in per_k.groupby(["dataset", "method"]):
        g = g.sort_values("k")
        ks = g["k"].to_numpy(dtype=float)
        scores = g["score"].to_numpy(dtype=float)
        if len(ks) == 1:
            auc = float(scores[0])
        else:
            auc = float(np.trapezoid(scores, ks) / (ks.max() - ks.min()))
        records.append({"dataset": dataset, "method": method, "auc_k": auc})
    tidy = pd.DataFrame.from_records(records)
    return tidy.pivot(index="dataset", columns="method", values="auc_k")


def mean_ranks(score_matrix: pd.DataFrame) -> pd.Series:
    """Average rank per method across datasets (rank 1 = highest score)."""
    ranks = score_matrix.rank(axis=1, ascending=False)
    return ranks.mean(axis=0).sort_values()


def method_rank_table(results: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per-dataset ranks with a trailing ``mean_rank`` row (best-first columns)."""
    scores = auc_k_table(results, task)
    ranks = scores.rank(axis=1, ascending=False)
    ranks.loc["mean_rank"] = ranks.mean(axis=0)
    order = ranks.loc["mean_rank"].sort_values().index
    return ranks[order]
