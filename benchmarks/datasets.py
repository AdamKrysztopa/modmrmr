"""Dataset registry and loader for the OpenMRMR benchmark.

Every entry is ``name -> {loader, task, n, p, source, rationale, dependence}``.
``dependence`` is one of ``"linear"``, ``"nonlinear"``, ``"mixed"``, ``"unknown"`` and is a
best-effort tag of the dominant relevance/redundancy mechanism in the dataset;
it does not affect loading. ``loader`` is a
zero-arg callable returning ``(X: pd.DataFrame, y: pd.Series)``. Small sklearn-builtin
and synthetic sets load eagerly; OpenML / UCI / scikit-feature microarray sets are
registered with a loader but are NEVER downloaded by the unit tests (see the
``network`` pytest marker). Microarray ``.mat`` files and UCI CSVs live under
``benchmarks/data/``; :func:`benchmarks.fetch.ensure_dataset_file` downloads and
materialises them on first use, checksum-verified against pinned sources. Run
``uv run python -m benchmarks.fetch`` to provision them all ahead of an offline run.

OpenML ids are verified against the OpenML registry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    make_classification,
    make_friedman1,
)

from benchmarks.fetch import ensure_dataset_file

_DATA_DIR = Path(__file__).parent / "data"
_SEED = 0


# --------------------------------------------------------------------------- #
# Loader factories
# --------------------------------------------------------------------------- #
def _from_sklearn_bunch(
    fetch: Callable[..., object],
) -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _load() -> tuple[pd.DataFrame, pd.Series]:
        bunch = fetch(as_frame=True)
        X = bunch.data.reset_index(drop=True)
        y = pd.Series(np.asarray(bunch.target), name="target").reset_index(drop=True)
        return X, y

    return _load


def _make_friedman1_loader() -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _load() -> tuple[pd.DataFrame, pd.Series]:
        X_arr, y_arr = make_friedman1(n_samples=500, n_features=10, noise=0.1, random_state=_SEED)
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
        return X, pd.Series(y_arr, name="target")

    return _load


def _make_classification_loader() -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _load() -> tuple[pd.DataFrame, pd.Series]:
        X_arr, y_arr = make_classification(
            n_samples=600,
            n_features=50,
            n_informative=10,
            n_redundant=10,
            n_repeated=0,
            random_state=_SEED,
        )
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
        return X, pd.Series(y_arr, name="target")

    return _load


def _openml_loader(data_id: int) -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _load() -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import fetch_openml  # local import: network path

        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        X = pd.get_dummies(bunch.data, drop_first=True).reset_index(drop=True)
        y = pd.Series(np.asarray(bunch.target), name="target").reset_index(drop=True)
        return X, y

    return _load


def _skfeature_mat_loader(filename: str) -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _load() -> tuple[pd.DataFrame, pd.Series]:
        from scipy.io import loadmat  # local import

        path = ensure_dataset_file(filename, _DATA_DIR)
        mat = loadmat(path)
        X_arr = np.asarray(mat["X"], dtype=float)
        y_arr = np.asarray(mat["Y"]).ravel().astype(int)
        X = pd.DataFrame(X_arr, columns=[f"g{i}" for i in range(X_arr.shape[1])])
        return X, pd.Series(y_arr, name="target")

    _load.local_file = filename
    return _load


def _uci_csv_loader(filename: str, target_col: str) -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _load() -> tuple[pd.DataFrame, pd.Series]:
        path = ensure_dataset_file(filename, _DATA_DIR)
        df = pd.read_csv(path)
        y = df[target_col].reset_index(drop=True)
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True).reset_index(drop=True)
        return X, y

    _load.local_file = filename
    return _load


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
DATASETS: dict[str, dict] = {
    # ---- Classification (12) ------------------------------------------------
    "breast_cancer": {
        "loader": _from_sklearn_bunch(load_breast_cancer),
        "task": "classification",
        "n": 569,
        "p": 30,
        "source": "sklearn",
        "rationale": "Low-dim smoke test; correlated radius/perimeter/area exercise redundancy.",
        "dependence": "unknown",
    },
    "synthetic_clf": {
        "loader": _make_classification_loader(),
        "task": "classification",
        "n": 600,
        "p": 50,
        "source": "synthetic",
        "rationale": "Ground-truth informative+redundant features for CI-fast redundancy checks.",
        "dependence": "linear",
    },
    "spambase": {
        "loader": _openml_loader(44),
        "task": "classification",
        "n": 4601,
        "p": 57,
        "source": "openml:44",
        "rationale": "Mid-dim n>>p; correlated word/char-frequency features.",
        "dependence": "mixed",
    },
    "isolet": {
        "loader": _openml_loader(300),
        "task": "classification",
        "n": 1560,
        "p": 617,
        "source": "openml:300",
        "rationale": "Many features, 26-class, moderate redundancy.",
        "dependence": "unknown",
    },
    "madelon": {
        "loader": _openml_loader(1485),
        "task": "classification",
        "n": 2600,
        "p": 500,
        "source": "openml:1485",
        "rationale": "NIPS-2003 ground truth: 5 informative + 15 redundant + 480 probe.",
        "dependence": "nonlinear",
    },
    "gisette": {
        "loader": _openml_loader(41026),
        "task": "classification",
        "n": 7000,
        "p": 5000,
        "source": "openml:41026",
        "rationale": "High-dim NIPS-2003 with injected distractor features.",
        "dependence": "nonlinear",
    },
    "arcene": {
        "loader": _openml_loader(1458),
        "task": "classification",
        "n": 200,
        "p": 10000,
        "source": "openml:1458",
        "rationale": "p>>n mass-spec NIPS-2003; real + probe features.",
        "dependence": "nonlinear",
    },
    "colon": {
        "loader": _skfeature_mat_loader("colon.mat"),
        "task": "classification",
        "n": 62,
        "p": 2000,
        "source": "scikit-feature",
        "rationale": "Classic MRMR microarray (Alon 1999); p>>n.",
        "dependence": "unknown",
    },
    "allaml": {
        "loader": _skfeature_mat_loader("ALLAML.mat"),
        "task": "classification",
        "n": 72,
        "p": 7129,
        "source": "scikit-feature",
        "rationale": "Classic MRMR leukemia microarray (Golub 1999); p>>n.",
        "dependence": "unknown",
    },
    "lymphoma": {
        "loader": _skfeature_mat_loader("lymphoma.mat"),
        "task": "classification",
        "n": 96,
        "p": 4026,
        "source": "scikit-feature",
        "rationale": "Classic MRMR multiclass microarray; p>>n.",
        "dependence": "unknown",
    },
    "prostate_ge": {
        "loader": _skfeature_mat_loader("Prostate_GE.mat"),
        "task": "classification",
        "n": 102,
        "p": 5966,
        "source": "scikit-feature",
        "rationale": "Gene-expression p>>n; high feature correlation.",
        "dependence": "unknown",
    },
    "smk_can_187": {
        "loader": _skfeature_mat_loader("SMK_CAN_187.mat"),
        "task": "classification",
        "n": 187,
        "p": 19993,
        "source": "scikit-feature",
        "rationale": "Ultra-high-dim (p~20k) smoker/cancer expression.",
        "dependence": "unknown",
    },
    # ---- Regression (12) ----------------------------------------------------
    "diabetes": {
        "loader": _from_sklearn_bunch(load_diabetes),
        "task": "regression",
        "n": 442,
        "p": 10,
        "source": "sklearn",
        "rationale": "Low-dim regression baseline; some correlated features.",
        "dependence": "unknown",
    },
    "friedman1": {
        "loader": _make_friedman1_loader(),
        "task": "regression",
        "n": 500,
        "p": 10,
        "source": "synthetic",
        "rationale": "Ground-truth relevance: only 5 of 10 features drive y.",
        "dependence": "nonlinear",
    },
    "california_housing": {
        "loader": _from_sklearn_bunch(fetch_california_housing),
        "task": "regression",
        "n": 20640,
        "p": 8,
        "source": "sklearn",
        "rationale": "Large-n low-dim; correlated geo/income features.",
        "dependence": "unknown",
    },
    "wine_quality": {
        "loader": _openml_loader(287),
        "task": "regression",
        "n": 4898,
        "p": 11,
        "source": "openml:287",
        "rationale": "Mid-dim physicochemical; moderate collinearity.",
        "dependence": "unknown",
    },
    "cpu_activity": {
        "loader": _openml_loader(44978),
        "task": "regression",
        "n": 8192,
        "p": 21,
        "source": "openml:44978",
        "rationale": "CTR23; correlated system-activity counters.",
        "dependence": "unknown",
    },
    "moneyball": {
        "loader": _openml_loader(41021),
        "task": "regression",
        "n": 1232,
        "p": 14,
        "source": "openml:41021",
        "rationale": "CTR23; mixed types, missing values, small-n tabular.",
        "dependence": "unknown",
    },
    "house_prices": {
        "loader": _openml_loader(42165),
        "task": "regression",
        "n": 1460,
        "p": 80,
        "source": "openml:42165",
        "rationale": "Rich mixed-type tabular; many redundant house attributes.",
        "dependence": "mixed",
    },
    "communities_crime": {
        "loader": _openml_loader(46286),
        "task": "regression",
        "n": 1994,
        "p": 127,
        "source": "openml:46286",
        "rationale": "Many correlated socio-economic predictors; high collinearity.",
        "dependence": "mixed",
    },
    "superconductivity": {
        "loader": _openml_loader(43174),
        "task": "regression",
        "n": 21263,
        "p": 81,
        "source": "openml:43174",
        "rationale": "Engineered mean/wtd-mean families with strong redundancy.",
        "dependence": "mixed",
    },
    "ct_slices": {
        "loader": _uci_csv_loader("ct_slices.csv", "reference"),
        "task": "regression",
        "n": 53500,
        "p": 385,
        "source": "uci:206",
        "rationale": "High feature count; spatially redundant histogram bins.",
        "dependence": "mixed",
    },
    "blog_feedback": {
        "loader": _uci_csv_loader("blogfeedback.csv", "target"),
        "task": "regression",
        "n": 60021,
        "p": 280,
        "source": "uci:304",
        "rationale": "Large-n high-p; many derived/redundant traffic features.",
        "dependence": "mixed",
    },
    "riboflavin": {
        "loader": _openml_loader(46983),
        "task": "regression",
        "n": 71,
        "p": 4088,
        "source": "openml:46983",
        "rationale": "p>>n regression analogue of the microarray sets (Buhlmann 2014).",
        "dependence": "unknown",
    },
}


def local_filename(name: str) -> str | None:
    """Name of the file under ``benchmarks/data/`` this dataset needs, if any.

    Returns ``None`` for datasets that load without a provisioned file (sklearn
    builtins, synthetic sets, OpenML). Lets callers pre-flight provisioning, and
    lets the tests assert every file-backed entry has a registered download.
    """
    return getattr(DATASETS[name]["loader"], "local_file", None)


def list_datasets(task: str | None = None) -> list[str]:
    if task is None:
        return list(DATASETS)
    return [name for name, entry in DATASETS.items() if entry["task"] == task]


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series, str]:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASETS)}")
    entry = DATASETS[name]
    X, y = entry["loader"]()
    return X, y, entry["task"]
