"""Full-factorial mechanism-suite runner: golden-set recovery + ranking + downstream.

Companion to ``mechanism.protocol``: that module grades named *cells*
(``benchmarks.cells.CELLS``), which confounds relevance family, redundancy
scorer, operator, and aggregation into a handful of hand-picked combinations.
This module drives the same recovery/downstream grading, plus a ranking
metric, from a :class:`mechanism.factorial.SelectorSpec` via
:func:`mechanism.factorial.build_selector` instead -- so Phase C/D of the
design-space study can hold three axes fixed and vary the fourth cleanly.

Mirrors ``mechanism.protocol``'s exact train/test (``run_mechanism_grid``) and
train/val/test (``run_validation_selected_grid``) split logic and seeding, so
there is no leakage and results are directly comparable to the committed
runners: selection is always fit on TRAIN only; recovery/ranking read the
train-fit ``selected_idx_``; downstream is always scored on the held-out
test split; the ``val_*`` stop modes pick their operating point on the
validation split, then re-select on the full train split before grading.

Per-cell errors (e.g. a scorer that raises on a degenerate column) produce a
single row with ``error`` set (a string) and every metric column ``NaN``,
rather than aborting the whole grid or silently passing.

:func:`run_downstream_only_grid` runs the identical stop-mode/split protocol
against ``benchmarks``-registry datasets, which carry no ``GroundTruth`` mask
-- recovery and ranking columns are always ``NaN`` there, and only
``downstream_score`` is populated. Both entry points share the per-dataset
selection+split loop (:func:`_run_grid_for_dataset`); the only difference is
whether a :class:`mechanism.ground_truth.GroundTruth` is available to grade
against.
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.model_selection import train_test_split

from benchmarks.datasets import DATASETS, load_dataset
from mechanism.datasets import load_mechanism_dataset
from mechanism.factorial import SelectorSpec, _resolve_relevance, build_selector
from mechanism.ground_truth import GroundTruth
from mechanism.protocol import _downstream, _val_score
from mechanism.recovery import ranking_scores, recovery

FACTORIAL_COLUMNS = [
    "dataset",
    "dependence",
    "task",
    "spec",
    "relevance",
    "redundancy",
    "operator",
    "aggregation",
    "stop_mode",
    "k",
    "score_threshold",
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

_METRIC_COLUMNS = (
    "precision",
    "recall",
    "f1",
    "redundancy_rate",
    "noise_rate",
    "average_precision",
    "roc_auc",
    "downstream_score",
)


def _nan_metrics() -> dict[str, float]:
    return dict.fromkeys(_METRIC_COLUMNS, float("nan"))


def _metrics_from_idx(
    idx: list[int], gt: GroundTruth | None, downstream_score: float
) -> dict[str, float]:
    """Grade ``idx`` against ``gt``, or return NaN recovery/ranking metrics if ``gt is None``.

    ``gt is None`` is how :func:`run_downstream_only_grid` opts out of recovery/ranking
    grading for benchmark datasets, which carry no ground-truth mask -- only
    ``downstream_score`` is populated in that case.
    """
    if gt is None:
        metrics = _nan_metrics()
        metrics["downstream_score"] = downstream_score
        return metrics
    rec = recovery(idx, gt)
    rank = ranking_scores(idx, gt)
    return {
        "precision": rec.precision,
        "recall": rec.recall,
        "f1": rec.f1,
        "redundancy_rate": rec.redundancy_rate,
        "noise_rate": rec.noise_rate,
        "average_precision": rank.average_precision,
        "roc_auc": rank.roc_auc,
        "downstream_score": downstream_score,
    }


def _run_grid_for_dataset(
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
    """Run every (spec x seed) combination across all four stop modes for one dataset.

    Shared by :func:`run_factorial_grid` (``gt`` is a real :class:`GroundTruth`, recovery
    and ranking columns are populated) and :func:`run_downstream_only_grid` (``gt is
    None``, recovery/ranking columns are ``NaN``); the selection, splitting, and
    stop-mode logic is otherwise identical.
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
                    selector = build_selector(spec, task, k, None, seed)
                    selector.fit(X_train, y_train)
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
                        selector = build_selector(spec, task, None, t, seed)
                        selector.fit(X_train, y_train)
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
                    selector = build_selector(spec, task, k, None, seed)
                    selector.fit(X_tr, y_tr)
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
                    selector = build_selector(spec, task, best_k, None, seed)
                    selector.fit(X_train, y_train)
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
                        selector = build_selector(spec, task, None, t, seed)
                        selector.fit(X_tr, y_tr)
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
                        selector = build_selector(spec, task, None, best_t, seed)
                        selector.fit(X_train, y_train)
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


def run_factorial_grid(
    specs: list[SelectorSpec],
    datasets: list[str],
    ks: list[int],
    thresholds: list[float] | None,
    seeds: list[int],
    *,
    include_downstream: bool = True,
) -> pd.DataFrame:
    """Run every (dataset x spec x seed) combination across all four stop modes.

    ``stop_mode="fixed_k"`` and ``"threshold"`` sweep ``ks``/``thresholds`` on the
    outer train split and grade every point, mirroring
    :func:`mechanism.protocol.run_mechanism_grid`. ``stop_mode="val_fixed_k"`` and
    ``"val_threshold"`` pick the stopping hyperparameter on an inner validation
    split by downstream RF score, then re-select on the full outer train split and
    grade that single point, mirroring
    :func:`mechanism.protocol.run_validation_selected_grid`. ``threshold``/
    ``val_threshold`` rows are only emitted when ``thresholds`` is non-empty.

    Set ``include_downstream=False`` to skip the (relatively expensive) held-out
    RF downstream scoring and leave ``downstream_score`` as ``NaN`` -- the
    validation-selection stages still fit an RF internally to choose the
    stopping point, since that scoring is intrinsic to the val_* protocol, not
    optional reporting.
    """
    rows: list[dict] = []
    for dataset in datasets:
        X, y, task, gt = load_mechanism_dataset(dataset)
        rows.extend(
            _run_grid_for_dataset(
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


def run_downstream_only_grid(
    specs: list[SelectorSpec],
    datasets: list[str],
    ks: list[int],
    thresholds: list[float] | None,
    seeds: list[int],
) -> pd.DataFrame:
    """Run the identical stop-mode/split protocol against ``benchmarks``-registry datasets.

    Benchmark datasets (:data:`benchmarks.datasets.DATASETS`) carry no
    :class:`mechanism.ground_truth.GroundTruth` mask, so ``precision``/``recall``/
    ``f1``/``redundancy_rate``/``noise_rate``/``average_precision``/``roc_auc`` are
    always ``NaN`` here; only ``downstream_score`` (a held-out RF fit, exactly as in
    :func:`run_factorial_grid`) is populated. ``dependence`` is taken from the
    benchmark registry entry (``"benchmark"`` if the entry has none). Per-cell
    selection errors produce an error row, same convention as
    :func:`run_factorial_grid`. Returns a tidy DataFrame with columns ==
    ``FACTORIAL_COLUMNS``.
    """
    rows: list[dict] = []
    for dataset in datasets:
        X, y, task = load_dataset(dataset)
        dependence = DATASETS.get(dataset, {}).get("dependence", "benchmark")
        rows.extend(
            _run_grid_for_dataset(
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
