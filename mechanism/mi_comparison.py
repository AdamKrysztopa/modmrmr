"""MI-estimator comparison runner (Study 4): swap the mRMR relevance measure only.

Holds redundancy (``pearson_abs``), operator (``difference``), and aggregation
(``mean``) fixed and compares mutual-information estimators as the relevance
measure over the mechanism-suite golden datasets. Mirrors the leakage-free
train/test split used in :mod:`mechanism.fast_factorial_protocol`, but simpler:
fixed-``k`` only, no validation split, no memo cache (one MRMR fit per cell).
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.model_selection import train_test_split

from mechanism.datasets import load_mechanism_dataset
from mechanism.protocol import _downstream
from mechanism.recovery import ranking_scores, recovery
from modmrmr.core.estimator import MRMRSelector

MI_ESTIMATORS: tuple[str, ...] = (
    "mi_reg_k3",
    "mi_reg_k5",
    "mi_reg_k8",
    "mi_reg_k16",
    "ksg_mi",
    "ksg_k16",
    "mixed_ksg",
    "ami_adaptive",
    "bspline_mi",
    "kde_mi",
    "copula_mi",
    "gcmi",
)

MI_COMPARISON_COLUMNS: list[str] = [
    "dataset",
    "dependence",
    "task",
    "estimator",
    "k",
    "seed",
    "precision",
    "recall",
    "f1",
    "redundancy_rate",
    "noise_rate",
    "average_precision",
    "roc_auc",
    "downstream_score",
    "runtime_s",
    "error",
]

_METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "redundancy_rate",
    "noise_rate",
    "average_precision",
    "roc_auc",
    "downstream_score",
]


def _resolve_estimator(name: str, task: str) -> str:
    """Remap ``mi_reg_k{n}`` to ``mi_classif_k{n}`` on classification tasks only.

    Every other estimator name is used verbatim for both tasks (they either
    handle discrete labels natively or intentionally treat them as continuous
    — that is the point of the comparison).
    """
    if task == "classification" and name.startswith("mi_reg_"):
        return "mi_classif_" + name.removeprefix("mi_reg_")
    return name


def run_mi_comparison_grid(
    estimators: list[str],
    datasets: list[str],
    ks: list[int],
    seeds: list[int],
    *,
    include_downstream: bool = True,
) -> pd.DataFrame:
    """Run every (dataset x seed x estimator x k) cell, comparing MI estimators.

    Redundancy=``pearson_abs``, operator=``difference``, aggregation=``mean`` are
    fixed; only the relevance (MI estimator) measure varies. Selection is fit on
    a leakage-free train split; recovery/ranking are graded against the train-fit
    ``selected_idx_`` (the ordered pick list, used for BOTH ``recovery`` and
    ``ranking_scores``); downstream score (if requested) comes from the held-out
    test split. The ``estimator`` column records the ORIGINAL (un-resolved) name
    so the frame groups cleanly by the comparison axis even though classification
    datasets internally use the ``mi_classif_k*`` variant. Any exception raised
    while fitting/scoring a cell is captured as a single row with ``error`` set
    and all metric columns NaN — never fatal. Returns a DataFrame with columns
    == ``MI_COMPARISON_COLUMNS`` (empty frame with those columns if no rows).
    """
    rows: list[dict] = []
    for d_idx, dataset in enumerate(datasets, start=1):
        X, y, task, gt = load_mechanism_dataset(dataset)
        dependence = gt.dependence
        is_clf = task == "classification"
        n_features = X.shape[1]
        print(
            f"[run_mi_comparison_grid {d_idx}/{len(datasets)}] {dataset} "
            f"({len(estimators)} estimators x {len(ks)} ks x {len(seeds)} seeds)",
            flush=True,
        )
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=y if is_clf else None,
            )
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            for estimator in estimators:
                resolved = _resolve_estimator(estimator, task)
                for k in ks:
                    if k > n_features:
                        continue
                    common = {
                        "dataset": dataset,
                        "dependence": dependence,
                        "task": task,
                        "estimator": estimator,
                        "k": k,
                        "seed": seed,
                    }
                    t0 = time.perf_counter()
                    try:
                        selector = MRMRSelector(
                            n_features=k,
                            relevance=resolved,
                            redundancy="pearson_abs",
                            operator="difference",
                            aggregation="mean",
                            task=task,
                            random_state=seed,
                        )
                        selector.fit(X_train, y_train)
                        idx = list(selector.selected_idx_)
                        rec = recovery(idx, gt)
                        rank = ranking_scores(idx, gt)
                        downstream = (
                            _downstream(X_train, X_test, y_train, y_test, idx, task)
                            if include_downstream
                            else float("nan")
                        )
                        runtime_s = time.perf_counter() - t0
                        rows.append(
                            {
                                **common,
                                "precision": rec.precision,
                                "recall": rec.recall,
                                "f1": rec.f1,
                                "redundancy_rate": rec.redundancy_rate,
                                "noise_rate": rec.noise_rate,
                                "average_precision": rank.average_precision,
                                "roc_auc": rank.roc_auc,
                                "downstream_score": downstream,
                                "runtime_s": runtime_s,
                                "error": None,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - per-cell error capture, not fatal
                        runtime_s = time.perf_counter() - t0
                        rows.append(
                            {
                                **common,
                                **{col: float("nan") for col in _METRIC_COLUMNS},
                                "runtime_s": runtime_s,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

    if not rows:
        return pd.DataFrame(columns=MI_COMPARISON_COLUMNS)
    return pd.DataFrame(rows)[MI_COMPARISON_COLUMNS]
