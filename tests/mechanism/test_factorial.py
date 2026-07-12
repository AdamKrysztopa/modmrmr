"""Tests for the full-factorial SelectorSpec + build_selector engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mechanism.factorial import (
    AGGREGATIONS,
    CANONICAL_NAMED,
    FULL_FACTORIAL,
    OPERATORS,
    REDUNDANCY_SCORERS,
    RELEVANCE_FAMILIES,
    SelectorSpec,
    build_selector,
)
from modmrmr.core.estimator import MRMRSelector


def test_full_factorial_size_and_unique_labels() -> None:
    assert len(FULL_FACTORIAL) == 180
    assert (
        len(RELEVANCE_FAMILIES) * len(REDUNDANCY_SCORERS) * len(OPERATORS) * len(AGGREGATIONS)
        == 180
    )
    labels = [spec.label for spec in FULL_FACTORIAL]
    assert len(set(labels)) == 180


def test_selector_spec_label_format() -> None:
    spec = SelectorSpec(
        relevance_family="f",
        redundancy="pearson_abs",
        operator="difference",
        aggregation="mean",
    )
    assert spec.label == "f|pearson_abs|difference|mean"


def test_selector_spec_is_frozen() -> None:
    spec = FULL_FACTORIAL[0]
    with pytest.raises(FrozenInstanceError):
        spec.operator = "quotient"  # type: ignore[misc]


def test_build_selector_matches_operator_and_aggregation() -> None:
    spec = FULL_FACTORIAL[0]
    selector = build_selector(spec, "classification", 5, None, 0)
    assert isinstance(selector, MRMRSelector)
    assert selector.operator == spec.operator
    assert selector.aggregation == spec.aggregation
    assert selector.redundancy == spec.redundancy
    assert selector.n_features == 5
    assert selector.score_threshold is None
    assert selector.random_state == 0


def test_build_selector_resolves_f_family_by_task() -> None:
    spec = SelectorSpec("f", "pearson_abs", "difference", "mean")
    clf_selector = build_selector(spec, "classification", 5, None, 0)
    reg_selector = build_selector(spec, "regression", 5, None, 0)
    assert clf_selector.relevance == "f_classif"
    assert reg_selector.relevance == "f_regression"


def test_build_selector_resolves_mi_family_by_task() -> None:
    spec = SelectorSpec("mi", "mutual_info_sklearn", "difference", "mean")
    clf_selector = build_selector(spec, "classification", 5, None, 0)
    reg_selector = build_selector(spec, "regression", 5, None, 0)
    assert clf_selector.relevance == "mutual_info_classif"
    assert reg_selector.relevance == "mutual_info_regression"


def test_modmrmr_spec_builds_with_multiplicative_max() -> None:
    spec = CANONICAL_NAMED["ModMRMR"]
    selector = build_selector(spec, "classification", 5, None, 0)
    assert selector.operator == "multiplicative"
    assert selector.aggregation == "max"


def test_canonical_named_labels() -> None:
    assert set(CANONICAL_NAMED) == {"MID", "MIQ", "FCD", "FCQ", "ModMRMR"}
    assert CANONICAL_NAMED["MID"].operator == "difference"
    assert CANONICAL_NAMED["MIQ"].operator == "quotient"
    assert CANONICAL_NAMED["FCD"].operator == "difference"
    assert CANONICAL_NAMED["FCQ"].operator == "quotient"
    assert CANONICAL_NAMED["ModMRMR"].operator == "multiplicative"
    assert CANONICAL_NAMED["ModMRMR"].aggregation == "max"


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("family", list(RELEVANCE_FAMILIES))
def test_build_selector_fits_for_every_relevance_family(task: str, family: str) -> None:
    """Every relevance family must resolve to something that actually fits."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    if task == "classification":
        y = pd.Series((X["x0"] + X["x1"] > 0).astype(int))
    else:
        y = pd.Series(X["x0"] * 2 - X["x1"] + rng.normal(scale=0.1, size=n))

    spec = SelectorSpec(family, "pearson_abs", "difference", "mean")
    selector = build_selector(spec, task, 3, None, 0)
    selector.fit(X, y)
    assert len(selector.selected_idx_) == 3


def test_build_selector_unknown_task_raises() -> None:
    spec = FULL_FACTORIAL[0]
    with pytest.raises(ValueError):
        build_selector(spec, "auto", 5, None, 0)
