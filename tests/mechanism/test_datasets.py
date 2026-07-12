import pandas as pd
import pytest

from mechanism.datasets import (
    MECHANISM_DATASETS,
    list_mechanism_datasets,
    load_mechanism_dataset,
)
from mechanism.ground_truth import GroundTruth

_SYNTHETIC = {
    "parabola",
    "radial",
    "sine",
    "nonlin_redundancy_clf",
    "nonlin_redundancy_reg",
    "linear_control_clf",
    "linear_control_reg",
    "mixed",
    "xor",
    "redundant_blocks_clf",
    "redundant_blocks_reg",
    "quotient_trap_reg",
    "quotient_trap_clf",
    "checkerboard",
    "annulus",
    "multiplicative_interaction",
    "threshold_and",
    "saturation",
    "ratio",
    "max_of",
    "heteroscedastic",
    "sinc_2d",
    "polynomial_sign",
    "conditional_redundancy",
}
_PROBES = {"probe_breast_cancer", "probe_diabetes"}


def test_registry_has_all_synthetic_and_probe_sets():
    names = set(MECHANISM_DATASETS)
    assert _SYNTHETIC <= names
    assert _PROBES <= names
    assert len(names) == len(_SYNTHETIC) + len(_PROBES)


def test_list_mechanism_datasets_matches_registry():
    assert set(list_mechanism_datasets()) == set(MECHANISM_DATASETS)


def test_every_registry_entry_has_valid_task_and_dependence():
    for name, entry in MECHANISM_DATASETS.items():
        assert entry["task"] in {"classification", "regression"}, name
        assert entry["dependence"] in {"linear", "nonlinear", "mixed"}, name
        assert callable(entry["loader"]), name


@pytest.mark.parametrize("name", sorted(_SYNTHETIC))
def test_load_synthetic_dataset_shapes_match_ground_truth(name):
    X, y, task, gt = load_mechanism_dataset(name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(gt, GroundTruth)
    assert gt.n_features == X.shape[1]
    assert len(y) == len(X)
    assert task in {"classification", "regression"}


def test_load_parabola_returns_expected_task_and_dependence():
    X, y, task, gt = load_mechanism_dataset("parabola")
    assert gt.n_features == X.shape[1]
    assert task == "regression"
    assert gt.dependence == "nonlinear"


@pytest.mark.parametrize("name", sorted(_PROBES))
def test_load_probe_dataset_shapes_match_ground_truth(name):
    X, y, task, gt = load_mechanism_dataset(name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert gt.n_features == X.shape[1]
    assert len(y) == len(X)
    assert len(gt.noise) == 10
    assert task in {"classification", "regression"}


def test_load_probe_breast_cancer_is_classification():
    _, _, task, _ = load_mechanism_dataset("probe_breast_cancer")
    assert task == "classification"


def test_load_probe_diabetes_is_regression():
    _, _, task, _ = load_mechanism_dataset("probe_diabetes")
    assert task == "regression"


def test_load_quotient_trap_reg_returns_expected_task_and_dependence():
    X, y, task, gt = load_mechanism_dataset("quotient_trap_reg")
    assert gt.n_features == X.shape[1]
    assert task == "regression"
    assert gt.dependence == "linear"


def test_load_quotient_trap_clf_returns_expected_task_and_dependence():
    X, y, task, gt = load_mechanism_dataset("quotient_trap_clf")
    assert gt.n_features == X.shape[1]
    assert task == "classification"
    assert gt.dependence == "linear"


def test_unknown_dataset_name_raises_with_helpful_message():
    with pytest.raises(KeyError, match="unknown mechanism dataset"):
        load_mechanism_dataset("does_not_exist")


def test_load_mechanism_dataset_deterministic():
    X1, y1, task1, gt1 = load_mechanism_dataset("probe_breast_cancer")
    X2, y2, task2, gt2 = load_mechanism_dataset("probe_breast_cancer")
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    assert task1 == task2
    assert gt1 == gt2
