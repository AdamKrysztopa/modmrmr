"""Evaluation protocol: strict in-fold selection, k-curves, stability, grid runner.

Leakage discipline: feature selection AND scaling are fit only on the training fold.
Seeds are explicit everywhere. Task detection is a local heuristic so this module
never depends on Plan A.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from benchmarks.datasets import DATASETS, load_dataset

# Canonical 17-column results schema (00-interface-contract.md §CANONICAL RESULTS
# SCHEMA). Plan D's analysis.schema.RESULT_COLUMNS MUST equal this list, in this order.
RESULT_COLUMNS = [
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

# Per-task metric reported by run_grid (contract PRIMARY_METRIC).
_PRIMARY_METRIC = {"classification": "balanced_accuracy", "regression": "r2"}

_DEFAULT_KS = [1, 2, 5, 10, 20, 50]


def _detect_task(y) -> str:
    """Classification if non-numeric OR integer-valued with few uniques, else regression.

    Mirrors the interface-contract heuristic: non-numeric dtype OR (integer-valued AND
    n_unique <= max(20, 0.05*n)) -> classification.
    """
    y = pd.Series(y)
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    values = y.dropna().to_numpy()
    n = len(values)
    is_integer_valued = np.array_equal(values, np.round(values))
    if is_integer_valued and y.nunique() <= max(20, 0.05 * n):
        return "classification"
    return "regression"


def stability_index(Z: np.ndarray) -> float:
    """Nogueira, Sechidis, Brown (2018) stability estimator.

    Z is an M x d binary matrix: row i is the selection indicator of resample i.
    Let p_f = mean of column f, kbar = sum_f p_f (mean subset size), and the unbiased
    per-feature variance s_f^2 = (M/(M-1)) p_f (1 - p_f). Then

        Phi = 1 - ( (1/d) sum_f s_f^2 ) / ( (kbar/d)(1 - kbar/d) ).

    Phi = 1 for identical subsets, ~0 for random selection, <=1 always.
    """
    Z = np.asarray(Z, dtype=float)
    M, d = Z.shape
    if M < 2:
        raise ValueError("stability_index needs at least 2 resamples")
    p = Z.mean(axis=0)
    kbar = p.sum()
    denom = (kbar / d) * (1.0 - kbar / d)
    if denom == 0.0:
        return 1.0
    s2 = (M / (M - 1.0)) * p * (1.0 - p)
    return float(1.0 - s2.mean() / denom)


def stability(selector, X, y, n_boot, k) -> float:
    """Bootstrap the training set n_boot times, refit, and return the stability index.

    Deterministic: the bootstrap RNG is seeded with 0 so repeated calls on the same
    (selector, X, y, n_boot, k) return an identical value.
    """
    task = _detect_task(y)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    n, p = X.shape
    rng = np.random.default_rng(0)
    Z = np.zeros((n_boot, p), dtype=float)
    for b in range(n_boot):
        rows = rng.integers(0, n, size=n)
        idx = selector.select(X.iloc[rows], y.iloc[rows], task, k)
        Z[b, idx] = 1.0
    return stability_index(Z)


def default_learners(task: str) -> dict:
    """Cheap, diverse downstream estimators (model-agnostic filter evaluation)."""
    if task == "classification":
        return {
            "knn": KNeighborsClassifier(n_neighbors=5),
            "linear": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(n_estimators=100, random_state=0),
        }
    return {
        "knn": KNeighborsRegressor(n_neighbors=5),
        "linear": Ridge(),
        "rf": RandomForestRegressor(n_estimators=100, random_state=0),
    }


def _make_splitter(task: str, n_splits: int, seed: int):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _score(task: str, y_true, y_pred) -> float:
    if task == "classification":
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(r2_score(y_true, y_pred))


def _evaluate(selector, X, y, task, ks, learners, n_splits, seed) -> list[dict]:
    """Leakage-free CV: selection + scaling fit on the TRAIN fold only.

    Each row is ``{k, learner, fold, metric, score, runtime_s}`` where ``runtime_s`` is
    the wall-time of the (train-fold-only) selection call for that ``(fold, k)``.
    """
    if learners is None:
        learners = default_learners(task)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    metric_name = _PRIMARY_METRIC[task]
    splitter = _make_splitter(task, n_splits, seed)
    rows: list[dict] = []
    for fold, (tr, te) in enumerate(splitter.split(X.to_numpy(), y.to_numpy())):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        for k in ks:
            # ---- selection on TRAIN ONLY --------------------------------- #
            t0 = time.perf_counter()
            idx = selector.select(X_tr, y_tr, task, k)
            runtime_s = time.perf_counter() - t0
            cols = X_tr.columns[idx]
            # ---- scaling fit on TRAIN ONLY ------------------------------- #
            scaler = StandardScaler().fit(X_tr[cols].to_numpy())
            X_tr_s = scaler.transform(X_tr[cols].to_numpy())
            X_te_s = scaler.transform(X_te[cols].to_numpy())
            for lname, model in learners.items():
                fitted = clone(model).fit(X_tr_s, y_tr.to_numpy())
                pred = fitted.predict(X_te_s)
                rows.append(
                    {
                        "k": k,
                        "learner": lname,
                        "fold": fold,
                        "metric": metric_name,
                        "score": _score(task, y_te.to_numpy(), pred),
                        "runtime_s": float(runtime_s),
                    }
                )
    return rows


def k_curve(selector, X, y, task, ks, learners, cv) -> pd.DataFrame:
    """Downstream score vs. feature count k, as a tidy long-format DataFrame."""
    rows = _evaluate(selector, X, y, task, ks, learners, n_splits=cv, seed=0)
    return pd.DataFrame(rows, columns=["k", "learner", "fold", "metric", "score"])


def run_grid(cells, datasets, learners, cv, seeds, ks=None, n_boot=10) -> pd.DataFrame:
    """Run every (dataset x cell x seed) combination into one tidy long DataFrame.

    Returns the canonical 17-column schema (``RESULT_COLUMNS``). Feature counts default
    to ``_DEFAULT_KS``, each capped at the dataset's p by the adapters. Execution of the
    full published grid is a later phase; this function is the wiring that phase calls.
    """
    ks = _DEFAULT_KS if ks is None else ks
    frames: list[pd.DataFrame] = []
    for d_idx, name in enumerate(datasets, start=1):
        X, y, task = load_dataset(name)
        n_samples, n_features = X.shape
        print(
            f"[run_grid {d_idx}/{len(datasets)}] {name} "
            f"({len(cells)} cells x {len(seeds)} seeds x cv={cv})",
            flush=True,
        )
        # Prefer the registry's canonical (n, p) so metadata reflects the full dataset,
        # falling back to the loaded frame's shape for stubbed/synthetic datasets.
        entry = DATASETS.get(name, {})
        reg_n = entry.get("n", n_samples)
        reg_p = entry.get("p", n_features)
        effective_ks = [k for k in ks if k <= n_features] or [min(ks[0], n_features)]
        for cell in cells:
            relevance = ""
            resolver = getattr(cell, "_resolve_relevance", None)
            if callable(resolver):
                relevance = resolver(task)
            # Nogueira stability is a property of (selector, data, k) — its bootstrap
            # is seed-independent (contract-pinned signature has no seed), so compute
            # it once per (dataset, method, k) and reuse across seeds and fold rows.
            stab_by_k = {k: stability(cell, X, y, n_boot, k) for k in effective_ks}
            for seed in seeds:
                rows = _evaluate(cell, X, y, task, effective_ks, learners, n_splits=cv, seed=seed)
                frame = pd.DataFrame(rows)
                frame["stability"] = frame["k"].map(stab_by_k)
                frame["dataset"] = name
                frame["task"] = task
                frame["method"] = cell.name
                frame["operator"] = getattr(cell, "operator", "")
                frame["aggregation"] = getattr(cell, "aggregation", "")
                frame["relevance"] = relevance
                frame["redundancy"] = getattr(cell, "redundancy", "")
                frame["n_samples"] = reg_n
                frame["n_features"] = reg_p
                frame["seed"] = seed
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[RESULT_COLUMNS]
