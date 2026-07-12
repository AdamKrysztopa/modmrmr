"""Memoized fast full-factorial mechanism-suite runner.

Companion to :mod:`mechanism.factorial_protocol` (the reference oracle, left
untouched). That module's ``_run_grid_for_dataset`` builds a fresh
``MRMRSelector`` and refits for every ``(spec, k, threshold, val-candidate)``
combination. The expensive part of a fit -- the relevance vector and the p x p
redundancy matrix -- depends ONLY on ``(train split, relevance scorer,
redundancy scorer)``, never on ``operator``/``aggregation``/``k``/
``score_threshold``. A full-factorial grid has 5 relevance families x 4
redundancy scorers = 20 distinct measure-pairs, each reused by 9
operator x aggregation specs x every ``k``/``threshold``/val candidate, so the
committed runner recomputes the same relevance vector and redundancy matrix
dozens of times per ``(dataset, seed, split-role, relevance, redundancy)``.

This module computes each relevance vector / redundancy matrix ONCE per
``(dataset, seed, split-role, relevance_name, redundancy_name)``, caches it,
and drives ``MRMRSelector`` from the cache via a pair of passthrough
callables that hand back exactly what the estimator would have computed
itself -- so the greedy selection every row performs is byte-identical to the
committed runner, only computed cheaply. See
``tests/mechanism/test_fast_equivalence.py`` for the hard equivalence gate
against :func:`mechanism.factorial_protocol.run_factorial_grid`.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from benchmarks.datasets import DATASETS, load_dataset
from mechanism.datasets import load_mechanism_dataset
from mechanism.factorial import SelectorSpec, _resolve_relevance
from mechanism.factorial_protocol import FACTORIAL_COLUMNS, _metrics_from_idx, _nan_metrics
from mechanism.ground_truth import GroundTruth
from mechanism.protocol import _downstream, _val_score
from modmrmr.core.estimator import MRMRSelector
from modmrmr.core.estimator import _resolve_redundancy as _core_resolve_redundancy
from modmrmr.core.estimator import _resolve_relevance as _core_resolve_relevance

# cache key -> ("ok", relevance: pd.Series, redundancy: pd.DataFrame) | ("err", Exception)
_MeasuresCache = dict[tuple[Any, ...], tuple]


def _compute_measures(
    X: pd.DataFrame,
    y: pd.Series,
    relevance_name: str,
    redundancy_name: str,
    task: str,
    random_state: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute ``(relevance, redundancy)`` EXACTLY as ``MRMRSelector.fit`` would.

    Calls the identical private resolution helpers ``MRMRSelector.fit`` uses
    (``modmrmr.core.estimator._resolve_relevance``/``_resolve_redundancy``, then
    the instance method ``_resolve_random_state`` that conditionally binds
    ``random_state`` via ``functools.partial`` when the resolved callable's
    signature accepts it) on a throwaway selector instance, so this is not a
    reimplementation that could silently drift from ``fit``'s behavior -- it is
    the same code path, called once instead of once per row.
    """
    probe = MRMRSelector(random_state=random_state)
    imp_func = probe._resolve_random_state(
        _core_resolve_relevance(relevance_name, task, random_state)
    )
    pen_func = probe._resolve_random_state(
        _core_resolve_redundancy(redundancy_name, task, random_state)
    )
    relevance = pd.Series(imp_func(X, y), index=X.columns)
    redundancy = pen_func(X)
    return relevance, redundancy


def _get_measures(
    cache: _MeasuresCache,
    key: tuple[Any, ...],
    X: pd.DataFrame,
    y: pd.Series,
    relevance_name: str,
    redundancy_name: str,
    task: str,
    random_state: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Cached ``_compute_measures``; a computation failure is cached and re-raised.

    Mirrors the per-row semantics of the reference oracle exactly: every row
    that would have hit the same scorer failure inside its own ``selector.fit``
    call still raises (with the same exception type/message) rather than the
    fast path silently skipping the failure because an earlier row already hit
    it.
    """
    if key not in cache:
        try:
            rel, red = _compute_measures(X, y, relevance_name, redundancy_name, task, random_state)
            cache[key] = ("ok", rel, red)
        except Exception as exc:  # noqa: BLE001 - cached and replayed per-row, never fatal here
            cache[key] = ("err", exc)
    entry = cache[key]
    if entry[0] == "err":
        raise entry[1]
    return entry[1], entry[2]


def _relevance_passthrough(cached: pd.Series):
    """``(X, y, **kwargs) -> array``, identical to what a real relevance scorer returns."""

    def _fn(X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any:
        return cached.to_numpy()

    return _fn


def _redundancy_passthrough(cached: pd.DataFrame):
    """``(X, **kwargs) -> DataFrame``, identical to what a real redundancy scorer returns."""

    def _fn(X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return cached

    return _fn


def _fast_fit(
    cache: _MeasuresCache,
    key_prefix: tuple[Any, ...],
    X: pd.DataFrame,
    y: pd.Series,
    spec: SelectorSpec,
    relevance_name: str,
    task: str,
    n_features: int | None,
    score_threshold: float | None,
    seed: int,
) -> MRMRSelector:
    """Build + fit an ``MRMRSelector`` from cached relevance/redundancy for ``spec``.

    The measure-pair cache key is ``(*key_prefix, relevance_name, spec.redundancy)`` --
    ``key_prefix`` identifies the split (e.g. ``(seed, "train")`` vs. ``(seed, "tr")``),
    so a fit against a different split, or a different measure pair on the same split,
    never shares a cache entry.
    """
    key = (*key_prefix, relevance_name, spec.redundancy)
    rel, red = _get_measures(cache, key, X, y, relevance_name, spec.redundancy, task, seed)
    selector = MRMRSelector(
        n_features=n_features,
        relevance=_relevance_passthrough(rel),
        redundancy=_redundancy_passthrough(red),
        operator=spec.operator,
        aggregation=spec.aggregation,
        task=task,
        random_state=seed,
        score_threshold=score_threshold,
    )
    selector.fit(X, y)
    return selector


def _run_grid_for_dataset_fast(
    specs: list[SelectorSpec],
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    dataset: str,
    dependence: str,
    gt: GroundTruth | None,
    ks: list[int],
    thresholds: list[float] | None,
    seeds: list[int],
    *,
    include_downstream: bool,
) -> list[dict]:
    """Memoized twin of ``mechanism.factorial_protocol._run_grid_for_dataset``.

    Identical split construction, stop-mode logic, and row shape; the only
    difference is that ``build_selector(spec, ...).fit(...)`` is replaced by
    :func:`_fast_fit`, which reuses a per-``(seed, split-role, relevance,
    redundancy)`` cache instead of recomputing relevance/redundancy from
    scratch on every call.
    """
    rows: list[dict] = []
    n_features = X.shape[1]
    is_clf = task == "classification"
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y if is_clf else None
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.3,
            random_state=seed,
            stratify=y_train if is_clf else None,
        )
        X_tr = X_tr.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        # Two distinct split-roles get fit against within a (dataset, seed): the full
        # outer train split ("train": fixed_k/threshold rows and the val_* final
        # re-select) and the inner fit-only split ("tr": the val_* candidate sweep).
        # Cached separately since they are different data.
        measures_cache: _MeasuresCache = {}
        train_key = (seed, "train")
        tr_key = (seed, "tr")

        for spec in specs:
            relevance = _resolve_relevance(spec.relevance_family, task)
            base = {
                "dataset": dataset,
                "dependence": dependence,
                "task": task,
                "spec": spec.label,
                "relevance": relevance,
                "redundancy": spec.redundancy,
                "operator": spec.operator,
                "aggregation": spec.aggregation,
                "seed": seed,
            }

            # ---- fixed_k --------------------------------------------------- #
            for k in ks:
                if k > n_features:
                    continue
                t0 = time.perf_counter()
                row = {
                    **base,
                    "stop_mode": "fixed_k",
                    "k": k,
                    "score_threshold": float("nan"),
                }
                idx: list[int] | None = None
                try:
                    selector = _fast_fit(
                        measures_cache,
                        train_key,
                        X_train,
                        y_train,
                        spec,
                        relevance,
                        task,
                        k,
                        None,
                        seed,
                    )
                    idx = list(selector.selected_idx_)
                except Exception as exc:  # noqa: BLE001 - captured per-cell, never fatal
                    row.update(_nan_metrics())
                    row["error"] = f"{type(exc).__name__}: {exc}"
                if idx is not None:
                    downstream = (
                        _downstream(X_train, X_test, y_train, y_test, idx, task)
                        if include_downstream
                        else float("nan")
                    )
                    row.update(_metrics_from_idx(idx, gt, downstream))
                    row["error"] = None
                row["runtime_s"] = time.perf_counter() - t0
                rows.append(row)

            # ---- threshold --------------------------------------------------#
            if thresholds:
                for t in thresholds:
                    t0 = time.perf_counter()
                    row = {
                        **base,
                        "stop_mode": "threshold",
                        "k": float("nan"),
                        "score_threshold": t,
                    }
                    idx: list[int] | None = None
                    try:
                        selector = _fast_fit(
                            measures_cache,
                            train_key,
                            X_train,
                            y_train,
                            spec,
                            relevance,
                            task,
                            None,
                            t,
                            seed,
                        )
                        idx = list(selector.selected_idx_)
                        row["k"] = selector.n_selected_
                    except Exception as exc:  # noqa: BLE001
                        row.update(_nan_metrics())
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    if idx is not None:
                        downstream = (
                            _downstream(X_train, X_test, y_train, y_test, idx, task)
                            if include_downstream
                            else float("nan")
                        )
                        row.update(_metrics_from_idx(idx, gt, downstream))
                        row["error"] = None
                    row["runtime_s"] = time.perf_counter() - t0
                    rows.append(row)

            # ---- val_fixed_k ------------------------------------------------#
            k_candidates: list[tuple[float, int]] = []
            k_errors: list[str] = []
            runtime_s = 0.0
            for k in ks:
                if k > n_features:
                    continue
                t0 = time.perf_counter()
                try:
                    selector = _fast_fit(
                        measures_cache, tr_key, X_tr, y_tr, spec, relevance, task, k, None, seed
                    )
                    idx = list(selector.selected_idx_)
                    val_score = _val_score(X_tr, X_val, y_tr, y_val, idx, task)
                    k_candidates.append((val_score, k))
                except Exception as exc:  # noqa: BLE001 - a bad candidate is dropped
                    k_errors.append(f"k={k}: {type(exc).__name__}: {exc}")
                finally:
                    runtime_s += time.perf_counter() - t0

            row = {
                **base,
                "stop_mode": "val_fixed_k",
                "score_threshold": float("nan"),
            }
            if k_candidates:
                best_k = max(k_candidates, key=lambda c: (c[0], -c[1]))[1]
                row["k"] = best_k
                t0 = time.perf_counter()
                idx: list[int] | None = None
                try:
                    selector = _fast_fit(
                        measures_cache,
                        train_key,
                        X_train,
                        y_train,
                        spec,
                        relevance,
                        task,
                        best_k,
                        None,
                        seed,
                    )
                    idx = list(selector.selected_idx_)
                except Exception as exc:  # noqa: BLE001
                    row.update(_nan_metrics())
                    row["error"] = f"{type(exc).__name__}: {exc}"
                if idx is not None:
                    downstream = (
                        _downstream(X_train, X_test, y_train, y_test, idx, task)
                        if include_downstream
                        else float("nan")
                    )
                    row.update(_metrics_from_idx(idx, gt, downstream))
                    row["error"] = None
                runtime_s += time.perf_counter() - t0
            else:
                row["k"] = float("nan")
                row.update(_nan_metrics())
                detail = "; ".join(k_errors) if k_errors else "no k <= n_features"
                row["error"] = f"no valid val_fixed_k candidate: {detail}"
            row["runtime_s"] = runtime_s
            rows.append(row)

            # ---- val_threshold ----------------------------------------------#
            if thresholds:
                t_candidates: list[tuple[float, float]] = []
                t_errors: list[str] = []
                runtime_s_t = 0.0
                for t in thresholds:
                    t0 = time.perf_counter()
                    try:
                        selector = _fast_fit(
                            measures_cache, tr_key, X_tr, y_tr, spec, relevance, task, None, t, seed
                        )
                        idx = list(selector.selected_idx_)
                        val_score = _val_score(X_tr, X_val, y_tr, y_val, idx, task)
                        t_candidates.append((val_score, t))
                    except Exception as exc:  # noqa: BLE001 - a bad candidate is dropped
                        t_errors.append(f"t={t}: {type(exc).__name__}: {exc}")
                    finally:
                        runtime_s_t += time.perf_counter() - t0

                row = {
                    **base,
                    "stop_mode": "val_threshold",
                }
                if t_candidates:
                    best_t = max(t_candidates, key=lambda c: (c[0], c[1]))[1]
                    row["score_threshold"] = best_t
                    t0 = time.perf_counter()
                    idx: list[int] | None = None
                    try:
                        selector = _fast_fit(
                            measures_cache,
                            train_key,
                            X_train,
                            y_train,
                            spec,
                            relevance,
                            task,
                            None,
                            best_t,
                            seed,
                        )
                        idx = list(selector.selected_idx_)
                        row["k"] = selector.n_selected_
                    except Exception as exc:  # noqa: BLE001
                        row["k"] = float("nan")
                        row.update(_nan_metrics())
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    if idx is not None:
                        downstream = (
                            _downstream(X_train, X_test, y_train, y_test, idx, task)
                            if include_downstream
                            else float("nan")
                        )
                        row.update(_metrics_from_idx(idx, gt, downstream))
                        row["error"] = None
                    runtime_s_t += time.perf_counter() - t0
                else:
                    row["k"] = float("nan")
                    row["score_threshold"] = float("nan")
                    row.update(_nan_metrics())
                    detail = "; ".join(t_errors) if t_errors else "no threshold candidates"
                    row["error"] = f"no valid val_threshold candidate: {detail}"
                row["runtime_s"] = runtime_s_t
                rows.append(row)

    return rows


def run_fast_factorial_grid(
    specs: list[SelectorSpec],
    datasets: list[str],
    ks: list[int],
    thresholds: list[float] | None,
    seeds: list[int],
    *,
    include_downstream: bool = True,
) -> pd.DataFrame:
    """Memoized twin of :func:`mechanism.factorial_protocol.run_factorial_grid`.

    Produces numerically identical rows (see
    ``tests/mechanism/test_fast_equivalence.py``) by caching the relevance
    vector and redundancy matrix per ``(dataset, seed, split-role, relevance,
    redundancy)`` instead of recomputing them for every ``(spec, k,
    threshold)``. Same ``FACTORIAL_COLUMNS`` schema, same stop modes, same
    leakage-free splits as the reference oracle.
    """
    rows: list[dict] = []
    for dataset in datasets:
        X, y, task, gt = load_mechanism_dataset(dataset)
        rows.extend(
            _run_grid_for_dataset_fast(
                specs,
                X,
                y,
                task,
                dataset,
                gt.dependence,
                gt,
                ks,
                thresholds,
                seeds,
                include_downstream=include_downstream,
            )
        )

    if not rows:
        return pd.DataFrame(columns=FACTORIAL_COLUMNS)
    return pd.DataFrame(rows)[FACTORIAL_COLUMNS]


def run_fast_downstream_only_grid(
    specs: list[SelectorSpec],
    datasets: list[str],
    ks: list[int],
    thresholds: list[float] | None,
    seeds: list[int],
) -> pd.DataFrame:
    """Memoized twin of :func:`mechanism.factorial_protocol.run_downstream_only_grid`."""
    rows: list[dict] = []
    for dataset in datasets:
        X, y, task = load_dataset(dataset)
        dependence = DATASETS.get(dataset, {}).get("dependence", "benchmark")
        rows.extend(
            _run_grid_for_dataset_fast(
                specs,
                X,
                y,
                task,
                dataset,
                dependence,
                None,
                ks,
                thresholds,
                seeds,
                include_downstream=True,
            )
        )

    if not rows:
        return pd.DataFrame(columns=FACTORIAL_COLUMNS)
    return pd.DataFrame(rows)[FACTORIAL_COLUMNS]
