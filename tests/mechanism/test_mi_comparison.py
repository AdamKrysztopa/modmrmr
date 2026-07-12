"""Tests for ``mechanism.mi_comparison.run_mi_comparison_grid``.

Tiny grid: column/range guards, determinism, and neighbor-count-axis coverage.
Uses ``nonlin_redundancy_clf`` as the second (nonlinear) golden classification
dataset — confirmed present via ``list_mechanism_datasets()``.

NOTE (Study 4 finding): the runner deliberately does NOT hard-code a
Study-3-replication assertion. Study 3's "sklearn-n3 is the MI confound"
(n3 AP 0.625 → n8 1.0 on ``linear_control_clf``) was measured on a FULL-data
fit; under this runner's leakage-free 70/30 split the neighbor-count effect is
dataset-dependent and reverses on the easy linear set (n3 ties/beats n16). That
is a scientific result of the executed grid — not a property a unit test should
pin. These tests only guard the runner's contract (schema, ranges, determinism,
axis coverage).
"""

from __future__ import annotations

import pandas as pd

from mechanism.mi_comparison import MI_COMPARISON_COLUMNS, run_mi_comparison_grid

_ESTIMATORS = ["mi_reg_k3", "mi_reg_k16", "mixed_ksg"]
_DATASETS = ["linear_control_clf", "nonlin_redundancy_clf"]
_KS = [3, 5]
_SEEDS = [0]


def _run():
    return run_mi_comparison_grid(_ESTIMATORS, _DATASETS, _KS, _SEEDS)


def test_columns_match_schema():
    df = _run()
    assert list(df.columns) == MI_COMPARISON_COLUMNS


def test_recovery_and_ranking_scores_in_unit_range():
    df = _run()
    cols = [
        "precision",
        "recall",
        "f1",
        "redundancy_rate",
        "noise_rate",
        "average_precision",
        "roc_auc",
    ]
    for col in cols:
        values = df[col].dropna()
        assert not values.empty, f"expected at least one non-NaN value in {col}"
        assert (values >= 0.0).all() and (values <= 1.0).all(), f"{col} out of [0, 1]"


def test_deterministic_given_same_args():
    df1 = _run()
    df2 = _run()
    label_cols = [c for c in MI_COMPARISON_COLUMNS if c != "runtime_s"]
    pd.testing.assert_frame_equal(
        df1[label_cols].reset_index(drop=True),
        df2[label_cols].reset_index(drop=True),
    )


def test_neighbor_count_axis_is_covered_per_estimator():
    """The runner produces one row group per requested estimator on the axis.

    A structural contract check, NOT a scientific hypothesis (see module
    docstring): every requested estimator — including both ends of the
    neighbor-count axis, ``mi_reg_k3`` and ``mi_reg_k16`` — must appear in the
    output with real, in-range ranking-AP values on each dataset. Whether more
    neighbors help is a study result, not a unit-test invariant.
    """
    df = _run()
    for ds in _DATASETS:
        sub = df[df["dataset"] == ds]
        got = set(sub["estimator"].unique())
        assert got == set(_ESTIMATORS), f"{ds}: estimator axis {got} != {set(_ESTIMATORS)}"
        for est in _ESTIMATORS:
            ap = sub[sub["estimator"] == est]["average_precision"].dropna()
            assert not ap.empty, f"{ds}/{est}: no average_precision produced"
            assert (ap >= 0.0).all() and (ap <= 1.0).all()
