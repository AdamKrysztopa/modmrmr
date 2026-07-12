"""Wall-clock aggregation for the paper's cost analysis (spec section 2 item 8)."""

from __future__ import annotations

import pandas as pd

from mechanism.figures import measure_family


def runtime_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median wall-clock ``runtime_s`` per measure family, sorted ascending.

    Groups the factorial/MI schema (columns ``relevance``, ``redundancy``,
    ``runtime_s``) by :func:`mechanism.figures.measure_family` and returns a
    tidy DataFrame with columns ``["family", "mean_runtime_s", "median_runtime_s",
    "n_fits"]``, sorted by ``mean_runtime_s`` ascending.
    """
    work = df.assign(
        family=[
            measure_family(r, d) for r, d in zip(df["relevance"], df["redundancy"], strict=True)
        ]
    )
    out = (
        work.groupby("family")["runtime_s"]
        .agg(mean_runtime_s="mean", median_runtime_s="median", n_fits="count")
        .reset_index()
        .sort_values("mean_runtime_s", kind="stable")
        .reset_index(drop=True)
    )
    return out[["family", "mean_runtime_s", "median_runtime_s", "n_fits"]]


def runtime_vs_p(df: pd.DataFrame) -> pd.DataFrame:
    """Mean wall-clock ``runtime_s`` per (``operator``, ``p``), sorted by (operator, p).

    Groups the high-dim schema (columns ``p``, ``operator``, ``runtime_s``) and
    returns a tidy DataFrame with columns ``["p", "operator", "mean_runtime_s",
    "n_fits"]``, sorted by ``operator`` then ``p``.
    """
    out = (
        df.groupby(["operator", "p"])["runtime_s"]
        .agg(mean_runtime_s="mean", n_fits="count")
        .reset_index()
        .sort_values(["operator", "p"], kind="stable")
        .reset_index(drop=True)
    )
    return out[["p", "operator", "mean_runtime_s", "n_fits"]]
