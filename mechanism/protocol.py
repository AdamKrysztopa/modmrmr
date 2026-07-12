"""Mechanism-suite runner: grade cells against ground-truth recovery.

Separate from ``benchmarks.protocol``: this module selects features with each cell
on each mechanism dataset, scores the picks against the dataset's ``GroundTruth``
(precision/recall/f1/redundancy_rate/noise_rate via :func:`mechanism.recovery.recovery`),
and records a light downstream (held-out) score. It reuses the ``benchmarks`` cell
adapters and learners rather than reinventing them.
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from benchmarks.cells import CELLS, get_adapter
from benchmarks.protocol import _score, default_learners
from mechanism.datasets import load_mechanism_dataset
from mechanism.recovery import recovery

MECHANISM_COLUMNS = [
    "dataset",
    "dependence",
    "task",
    "method",
    "operator",
    "aggregation",
    "relevance",
    "redundancy",
    "stop_mode",
    "k",
    "score_threshold",
    "seed",
    "precision",
    "recall",
    "f1",
    "redundancy_rate",
    "noise_rate",
    "downstream_score",
    "runtime_s",
]


def _downstream(X_train, X_test, y_train, y_test, idx, task) -> float:
    if not idx:
        return float("nan")
    learner = clone(default_learners(task)["rf"])
    X_tr = X_train.iloc[:, idx]
    X_te = X_test.iloc[:, idx]
    learner.fit(X_tr, y_train)
    return _score(task, y_test, learner.predict(X_te))


def _val_score(X_train, X_val, y_train, y_val, idx, task) -> float:
    """RF fit/score used to grade a stopping-hyperparameter candidate on the val split.

    Unlike ``_downstream`` (which returns NaN for an empty selection so a downstream
    row can still be emitted), an empty ``idx`` here raises so the candidate is
    dropped from the val-score comparison entirely rather than silently scoring NaN.
    """
    if not idx:
        raise ValueError("empty selection cannot be scored on the validation split")
    learner = clone(default_learners(task)["rf"])
    learner.fit(X_train.iloc[:, idx], y_train)
    return _score(task, y_val, learner.predict(X_val.iloc[:, idx]))


def run_mechanism_grid(
    cells: list[str],
    datasets: list[str],
    ks: list[int],
    seeds: list[int],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Run every (dataset x cell x seed) combination, grading recovery + downstream.

    Fixed-k rows are emitted for every cell; threshold rows (``stop_mode="threshold"``)
    are emitted only when ``thresholds`` is not ``None`` and only for MRMR-family cells
    (``name in CELLS``) since baselines cannot threshold. Returns a tidy DataFrame with
    columns == ``MECHANISM_COLUMNS``.
    """
    rows: list[dict] = []
    for d_idx, dataset in enumerate(datasets, start=1):
        X, y, task, gt = load_mechanism_dataset(dataset)
        dependence = gt.dependence
        n_features = X.shape[1]
        print(
            f"[run_mechanism_grid {d_idx}/{len(datasets)}] {dataset} "
            f"({len(cells)} cells x {len(seeds)} seeds)",
            flush=True,
        )
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=y if task == "classification" else None,
            )
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            for name in cells:
                adapter = get_adapter(name)
                is_mrmr = name in CELLS
                operator = getattr(adapter, "operator", "")
                aggregation = getattr(adapter, "aggregation", "")
                redundancy = getattr(adapter, "redundancy", "")
                resolver = getattr(adapter, "_resolve_relevance", None)
                relevance = resolver(task) if callable(resolver) else ""

                for k in ks:
                    if k > n_features:
                        continue
                    t0 = time.perf_counter()
                    idx = adapter.select(X_train, y_train, task, k)
                    runtime_s = time.perf_counter() - t0
                    rec = recovery(idx, gt)
                    downstream = _downstream(X_train, X_test, y_train, y_test, idx, task)
                    rows.append(
                        {
                            "dataset": dataset,
                            "dependence": dependence,
                            "task": task,
                            "method": name,
                            "operator": operator,
                            "aggregation": aggregation,
                            "relevance": relevance,
                            "redundancy": redundancy,
                            "stop_mode": "fixed_k",
                            "k": k,
                            "score_threshold": float("nan"),
                            "seed": seed,
                            "precision": rec.precision,
                            "recall": rec.recall,
                            "f1": rec.f1,
                            "redundancy_rate": rec.redundancy_rate,
                            "noise_rate": rec.noise_rate,
                            "downstream_score": downstream,
                            "runtime_s": runtime_s,
                        }
                    )

                if is_mrmr and thresholds is not None:
                    for t in thresholds:
                        from modmrmr.core.estimator import MRMRSelector  # local import

                        t0 = time.perf_counter()
                        selector = MRMRSelector(
                            n_features=None,
                            relevance=relevance,
                            redundancy=redundancy,
                            operator=operator,
                            aggregation=aggregation,
                            task=task,
                            random_state=seed,
                            score_threshold=t,
                        )
                        selector.fit(X_train, y_train)
                        runtime_s = time.perf_counter() - t0
                        idx = list(selector.selected_idx_)
                        k_out = selector.n_selected_
                        rec = recovery(idx, gt)
                        downstream = _downstream(X_train, X_test, y_train, y_test, idx, task)
                        rows.append(
                            {
                                "dataset": dataset,
                                "dependence": dependence,
                                "task": task,
                                "method": name,
                                "operator": operator,
                                "aggregation": aggregation,
                                "relevance": relevance,
                                "redundancy": redundancy,
                                "stop_mode": "threshold",
                                "k": k_out,
                                "score_threshold": t,
                                "seed": seed,
                                "precision": rec.precision,
                                "recall": rec.recall,
                                "f1": rec.f1,
                                "redundancy_rate": rec.redundancy_rate,
                                "noise_rate": rec.noise_rate,
                                "downstream_score": downstream,
                                "runtime_s": runtime_s,
                            }
                        )

    if not rows:
        return pd.DataFrame(columns=MECHANISM_COLUMNS)
    return pd.DataFrame(rows)[MECHANISM_COLUMNS]


def run_validation_selected_grid(
    cells: list[str],
    datasets: list[str],
    ks: list[int],
    thresholds: list[float] | None,
    seeds: list[int],
) -> pd.DataFrame:
    """Grade a *validation-selected* stopping protocol: the real-world recipe.

    Instead of sweeping a fixed ``k`` or ``score_threshold`` and reporting every point
    on the sweep (as :func:`run_mechanism_grid` does), this picks the stopping
    hyperparameter itself on a held-out validation split by downstream RF performance,
    then reports, for that single chosen value, recovery on the full-train re-selection
    and the downstream score on the held-out test split (never a test-side selection).

    Per ``(dataset, seed)``: an outer 70/30 train/test split, then an inner 70/30
    split of the train portion into fit/validation. For every cell, ``stop_mode``
    ``"val_fixed_k"`` sweeps ``ks`` on ``(X_tr, y_tr)``, scores each candidate's RF
    downstream performance on ``X_val``, picks ``k*`` (max val score, ties -> smallest
    k), then re-selects at ``k*`` on the FULL train split and grades recovery/test
    downstream there. MRMR-family cells (``name in CELLS``) additionally get
    ``stop_mode="val_threshold"`` when ``thresholds`` is non-empty: same recipe over
    ``MRMRSelector(score_threshold=t)`` candidates, ties -> the larger (sparser) t.

    A val candidate that raises (including an empty selection, which cannot be RF
    scored) is dropped from the comparison rather than crashing the grid; if every
    candidate for a cell errors, that cell/dataset/seed combination yields no row.
    Returns a tidy DataFrame with columns == ``MECHANISM_COLUMNS``.
    """
    rows: list[dict] = []
    for d_idx, dataset in enumerate(datasets, start=1):
        X, y, task, gt = load_mechanism_dataset(dataset)
        dependence = gt.dependence
        n_features = X.shape[1]
        is_clf = task == "classification"
        print(
            f"[run_validation_selected_grid {d_idx}/{len(datasets)}] {dataset} "
            f"({len(cells)} cells x {len(seeds)} seeds)",
            flush=True,
        )
        for seed in seeds:
            X_tr_full, X_test, y_tr_full, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=y if is_clf else None,
            )
            X_tr_full = X_tr_full.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_tr_full = y_tr_full.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_full,
                y_tr_full,
                test_size=0.3,
                random_state=seed,
                stratify=y_tr_full if is_clf else None,
            )
            X_tr = X_tr.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True)
            y_tr = y_tr.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)

            for name in cells:
                adapter = get_adapter(name)
                is_mrmr = name in CELLS
                operator = getattr(adapter, "operator", "")
                aggregation = getattr(adapter, "aggregation", "")
                redundancy = getattr(adapter, "redundancy", "")
                resolver = getattr(adapter, "_resolve_relevance", None)
                relevance = resolver(task) if callable(resolver) else ""
                common = {
                    "dataset": dataset,
                    "dependence": dependence,
                    "task": task,
                    "method": name,
                    "operator": operator,
                    "aggregation": aggregation,
                    "relevance": relevance,
                    "redundancy": redundancy,
                    "seed": seed,
                }

                # ---- val_fixed_k (every cell) --------------------------------- #
                k_candidates: list[tuple[float, int]] = []
                runtime_s = 0.0
                for k in ks:
                    if k > n_features:
                        continue
                    t0 = time.perf_counter()
                    try:
                        idx = adapter.select(X_tr, y_tr, task, k)
                        val_score = _val_score(X_tr, X_val, y_tr, y_val, idx, task)
                    except Exception:  # noqa: BLE001 - a bad candidate is dropped, not fatal
                        continue
                    finally:
                        runtime_s += time.perf_counter() - t0
                    k_candidates.append((val_score, k))

                if k_candidates:
                    best_k = max(k_candidates, key=lambda c: (c[0], -c[1]))[1]
                    try:
                        t0 = time.perf_counter()
                        idx = adapter.select(X_tr_full, y_tr_full, task, best_k)
                        runtime_s += time.perf_counter() - t0
                        rec = recovery(idx, gt)
                        downstream = _downstream(X_tr_full, X_test, y_tr_full, y_test, idx, task)
                        rows.append(
                            {
                                **common,
                                "stop_mode": "val_fixed_k",
                                "k": best_k,
                                "score_threshold": float("nan"),
                                "precision": rec.precision,
                                "recall": rec.recall,
                                "f1": rec.f1,
                                "redundancy_rate": rec.redundancy_rate,
                                "noise_rate": rec.noise_rate,
                                "downstream_score": downstream,
                                "runtime_s": runtime_s,
                            }
                        )
                    except Exception:  # noqa: BLE001 - final re-select failing skips the row
                        pass

                # ---- val_threshold (MRMR-family only) ------------------------- #
                if is_mrmr and thresholds:
                    from modmrmr.core.estimator import MRMRSelector  # local import

                    t_candidates: list[tuple[float, float]] = []
                    runtime_s_t = 0.0
                    for t in thresholds:
                        t0 = time.perf_counter()
                        try:
                            selector = MRMRSelector(
                                n_features=None,
                                relevance=relevance,
                                redundancy=redundancy,
                                operator=operator,
                                aggregation=aggregation,
                                task=task,
                                random_state=seed,
                                score_threshold=t,
                            )
                            selector.fit(X_tr, y_tr)
                            idx = list(selector.selected_idx_)
                            val_score = _val_score(X_tr, X_val, y_tr, y_val, idx, task)
                        except Exception:  # noqa: BLE001 - a bad candidate is dropped, not fatal
                            continue
                        finally:
                            runtime_s_t += time.perf_counter() - t0
                        t_candidates.append((val_score, t))

                    if t_candidates:
                        best_t = max(t_candidates, key=lambda c: (c[0], c[1]))[1]
                        try:
                            t0 = time.perf_counter()
                            selector = MRMRSelector(
                                n_features=None,
                                relevance=relevance,
                                redundancy=redundancy,
                                operator=operator,
                                aggregation=aggregation,
                                task=task,
                                random_state=seed,
                                score_threshold=best_t,
                            )
                            selector.fit(X_tr_full, y_tr_full)
                            runtime_s_t += time.perf_counter() - t0
                            idx = list(selector.selected_idx_)
                            k_out = selector.n_selected_
                            rec = recovery(idx, gt)
                            downstream = _downstream(
                                X_tr_full, X_test, y_tr_full, y_test, idx, task
                            )
                            rows.append(
                                {
                                    **common,
                                    "stop_mode": "val_threshold",
                                    "k": k_out,
                                    "score_threshold": best_t,
                                    "precision": rec.precision,
                                    "recall": rec.recall,
                                    "f1": rec.f1,
                                    "redundancy_rate": rec.redundancy_rate,
                                    "noise_rate": rec.noise_rate,
                                    "downstream_score": downstream,
                                    "runtime_s": runtime_s_t,
                                }
                            )
                        except Exception:  # noqa: BLE001 - final refit failing skips the row
                            pass

    if not rows:
        return pd.DataFrame(columns=MECHANISM_COLUMNS)
    return pd.DataFrame(rows)[MECHANISM_COLUMNS]
