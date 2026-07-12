"""Tidy benchmark-results schema plus a deterministic synthetic fixture.

The schema is the Plan C ``run_grid`` output contract consumed by every
analysis generator. ``RESULT_COLUMNS`` mirrors ``benchmarks.protocol.RESULT_COLUMNS``
(the interface contract's CANONICAL RESULTS SCHEMA) verbatim — a test in
``tests/analysis/test_schema.py`` asserts the two stay identical.
``make_synthetic_results`` produces a small, fully populated table so the
generators can be validated without running the grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_COLUMNS: list[str] = [
    "dataset",
    "task",
    "method",
    "operator",
    "aggregation",
    "relevance",
    "redundancy",
    "n_samples",
    "n_features",
    "k",
    "learner",
    "seed",
    "fold",
    "metric",
    "score",
    "stability",
    "runtime_s",
]

PRIMARY_METRIC: dict[str, str] = {
    "classification": "balanced_accuracy",
    "regression": "r2",
}

# (method, operator, aggregation, relevance, redundancy, base_skill)
# base_skill biases synthetic scores so ranking is non-degenerate and testable.
_METHODS: list[tuple[str, str, str, str, str, float]] = [
    ("MID", "difference", "mean", "mutual_info", "pearson_abs", 0.02),
    ("MIQ", "quotient", "mean", "mutual_info", "pearson_abs", 0.00),
    ("FCD", "difference", "mean", "f_test", "pearson_abs", 0.03),
    ("FCQ", "quotient", "mean", "f_test", "pearson_abs", 0.01),
    ("ModMRMR", "multiplicative", "max", "f_test", "pearson_abs", 0.06),
    ("SelectKBest", "", "", "f_test", "", -0.02),
]

# (dataset, task, n_samples, n_features)
_DATASETS: list[tuple[str, str, int, int]] = [
    ("breast_cancer", "classification", 569, 30),
    ("madelon", "classification", 2600, 500),
    ("colon", "classification", 62, 2000),
    ("diabetes", "regression", 442, 10),
    ("superconduct", "regression", 21263, 81),
    ("riboflavin", "regression", 71, 4088),
]

_KS: tuple[int, ...] = (5, 10, 20)
_LEARNERS: tuple[str, ...] = ("knn", "rf")
_SEEDS: tuple[int, ...] = (0, 1)
_SECONDARY_METRIC: dict[str, str] = {"classification": "roc_auc", "regression": "rmse"}


def data_regime(n_samples: int, n_features: int) -> str:
    """Return the data regime label used to split the decision guidance."""
    return "p>>n" if n_features > n_samples else "n>=p"


def make_synthetic_results(seed: int = 0) -> pd.DataFrame:
    """Build a deterministic, schema-complete tidy results table for tests.

    Scores rise with ``k`` and a per-method ``base_skill`` so that AUC-over-k,
    ranks, stability, and the decision guide are all well defined and stable.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dataset, task, n_samples, n_features in _DATASETS:
        base_metric = 0.70 if task == "classification" else 0.45
        for method, operator, aggregation, relevance, redundancy, skill in _METHODS:
            stability = float(np.clip(0.55 + skill * 5.0, 0.0, 1.0))
            runtime = float(abs(rng.normal(1.0, 0.1)) * (1 + n_features / 1000))
            for k in _KS:
                k_gain = 0.05 * np.log1p(k)
                for learner in _LEARNERS:
                    for s in _SEEDS:
                        primary = float(
                            np.clip(
                                base_metric + skill + k_gain + rng.normal(0, 0.005),
                                0.0,
                                1.0,
                            )
                        )
                        for metric_name, value in (
                            (PRIMARY_METRIC[task], primary),
                            (_SECONDARY_METRIC[task], primary * 0.9),
                        ):
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "method": method,
                                    "operator": operator,
                                    "aggregation": aggregation,
                                    "relevance": relevance,
                                    "redundancy": redundancy,
                                    "n_samples": n_samples,
                                    "n_features": n_features,
                                    "k": k,
                                    "learner": learner,
                                    "seed": s,
                                    "fold": 0,
                                    "metric": metric_name,
                                    "score": value,
                                    "stability": stability,
                                    "runtime_s": runtime,
                                }
                            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
