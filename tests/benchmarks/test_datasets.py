import numpy as np
import pandas as pd
import pytest

from benchmarks.datasets import DATASETS, list_datasets, load_dataset

REQUIRED_KEYS = {"loader", "task", "n", "p", "source", "rationale"}


def test_registry_has_24_well_formed_entries():
    assert len(DATASETS) == 24
    n_clf = sum(1 for e in DATASETS.values() if e["task"] == "classification")
    n_reg = sum(1 for e in DATASETS.values() if e["task"] == "regression")
    assert n_clf == 12
    assert n_reg == 12
    for name, entry in DATASETS.items():
        assert REQUIRED_KEYS <= set(entry), name
        assert entry["task"] in {"classification", "regression"}, name
        assert callable(entry["loader"]), name
        assert isinstance(entry["p"], int) and entry["p"] > 0, name
        assert entry["rationale"], name


def test_list_datasets_filters_by_task():
    assert set(list_datasets("classification")) <= set(DATASETS)
    assert len(list_datasets("classification")) == 12
    assert len(list_datasets("regression")) == 12
    assert len(list_datasets()) == 24


@pytest.mark.parametrize("name", ["breast_cancer", "diabetes"])
def test_builtin_datasets_load_with_correct_shape_and_task(name):
    X, y, task = load_dataset(name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert task == DATASETS[name]["task"]
    assert X.shape[0] == len(y)
    assert X.shape[1] == DATASETS[name]["p"]
    assert X.shape[0] == DATASETS[name]["n"]


def test_synthetic_friedman1_has_ground_truth_shape():
    X, y, task = load_dataset("friedman1")
    assert task == "regression"
    assert X.shape == (500, 10)
    assert len(y) == 500
    assert np.isfinite(y.to_numpy()).all()


def test_synthetic_make_classification_loads():
    X, y, task = load_dataset("synthetic_clf")
    assert task == "classification"
    assert X.shape[1] == DATASETS["synthetic_clf"]["p"]
    assert set(np.unique(y.to_numpy())) <= {0, 1}


def test_openml_and_microarray_entries_are_registry_only():
    # These must NOT be downloaded in unit tests; assert metadata only.
    for name in ["madelon", "gisette", "colon", "riboflavin"]:
        entry = DATASETS[name]
        assert entry["source"].lower().startswith(("openml", "scikit-feature", "uci"))


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_dataset("does_not_exist")


# --------------------------------------------------------------------------- #
# Network-gated loader tests — one per remote loader factory (openml / UCI CSV /
# scikit-feature .mat). Deselected by default (`addopts` excludes `network`); run
# explicitly with `uv run pytest -m network`. Without these the registry's
# non-builtin loaders (14/24 datasets) have zero executed coverage.
# --------------------------------------------------------------------------- #
@pytest.mark.network
@pytest.mark.parametrize("name", ["spambase", "colon", "blog_feedback"])
def test_remote_loaders_return_wellformed_data(name):
    entry = DATASETS[name]
    X, y, task = load_dataset(name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert task == entry["task"]
    assert X.shape[0] == len(y)
    assert X.shape[0] > 0 and X.shape[1] > 0
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_every_entry_has_dependence_tag():
    valid = {"linear", "nonlinear", "mixed", "unknown"}
    for name, entry in DATASETS.items():
        assert entry.get("dependence") in valid, name
    # the known-nonlinear sets are tagged
    assert DATASETS["madelon"]["dependence"] in {"nonlinear", "mixed"}
    assert DATASETS["friedman1"]["dependence"] in {"nonlinear", "mixed"}
