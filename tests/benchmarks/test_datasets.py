import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from benchmarks import datasets
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
# Network-gated loader tests — one classification and one regression OpenML id,
# so a provider outage or a retired dataset id is diagnosed directly. Deselected
# by default (`addopts` excludes `network`); run explicitly with
# `uv run pytest -m network`.
#
# Only genuinely-remote loaders belong here. The file-backed loaders
# (scikit-feature .mat, UCI CSV) download nothing — they read hand-provisioned
# files out of `benchmarks/data/`, so marking them `network` made the scheduled
# evidence lane fail unconditionally on any machine without that directory. They
# are covered below against temporary fixture files instead.
# --------------------------------------------------------------------------- #
@pytest.mark.network
@pytest.mark.parametrize("name", ["spambase", "wine_quality"])
def test_remote_loaders_return_wellformed_data(name):
    entry = DATASETS[name]
    X, y, task = load_dataset(name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert task == entry["task"]
    assert X.shape[0] == len(y)
    assert X.shape[0] > 0 and X.shape[1] > 0
    assert np.isfinite(X.to_numpy(dtype=float)).all()


# --------------------------------------------------------------------------- #
# File-backed loader tests — the .mat and CSV parsing paths run against fixtures
# written into a temporary `_DATA_DIR`, giving these factories executed coverage
# without a download. Both loaders resolve `_DATA_DIR` at call time, so
# monkeypatching the module attribute redirects the registry entries too.
# --------------------------------------------------------------------------- #
@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_DATA_DIR", tmp_path)
    return tmp_path


def test_skfeature_mat_loader_reads_local_file(data_dir):
    savemat(
        data_dir / "colon.mat",
        {"X": np.arange(12, dtype=float).reshape(4, 3), "Y": np.array([[1], [2], [1], [2]])},
    )

    X, y, task = load_dataset("colon")

    assert task == "classification"
    assert X.shape == (4, 3)
    assert list(X.columns) == ["g0", "g1", "g2"]
    assert X.to_numpy().dtype == np.float64
    assert y.tolist() == [1, 2, 1, 2]


def test_uci_csv_loader_encodes_categoricals_and_splits_target(data_dir):
    pd.DataFrame(
        {"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"], "target": [10.0, 20.0, 30.0]}
    ).to_csv(data_dir / "blogfeedback.csv", index=False)

    X, y, task = load_dataset("blog_feedback")

    assert task == "regression"
    assert y.tolist() == [10.0, 20.0, 30.0]
    assert "target" not in X.columns
    # drop_first=True leaves one indicator for the two-level categorical.
    assert list(X.columns) == ["num", "cat_b"]
    assert X["cat_b"].tolist() == [False, True, False]


@pytest.mark.parametrize(
    ("name", "filename"), [("colon", "colon.mat"), ("blog_feedback", "blogfeedback.csv")]
)
def test_file_backed_loaders_report_missing_file_actionably(data_dir, name, filename):
    with pytest.raises(FileNotFoundError, match=filename):
        load_dataset(name)


def test_every_entry_has_dependence_tag():
    valid = {"linear", "nonlinear", "mixed", "unknown"}
    for name, entry in DATASETS.items():
        assert entry.get("dependence") in valid, name
    # the known-nonlinear sets are tagged
    assert DATASETS["madelon"]["dependence"] in {"nonlinear", "mixed"}
    assert DATASETS["friedman1"]["dependence"] in {"nonlinear", "mixed"}
