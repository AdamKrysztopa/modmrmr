"""External-baseline comparison runner (Task 7 / H5): mrmr-selection + skfeature vs modmrmr.

Runs external reference implementations (the smazzanti ``mrmr`` package,
skfeature's CMIM/JMI) alongside our canonical named modmrmr specs
(:data:`mechanism.factorial.CANONICAL_NAMED`) on the mechanism-suite golden
datasets, under the same leakage-free train/test split used elsewhere in the
mechanism suite. Grading recovery (F1 + noise_rate) against ground truth for
both external and internal methods on one shared grid is what lets the paper's
design-space claims be checked against something other than modmrmr itself.

Reuses, never reimplements: the external calls are not made directly here --
:data:`EXTERNAL_METHODS` wraps ``benchmarks.cells.MRMRSelectionBaseline`` and
the skfeature adapters already registered in ``benchmarks.cells.BASELINES``.
A modmrmr method name is any key of ``mechanism.factorial.CANONICAL_NAMED`` and
is run through :func:`mechanism.factorial.build_selector`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from analysis.highdim_study import recovery_metrics
from benchmarks.cells import BASELINES, MRMRSelectionBaseline
from mechanism.datasets import load_mechanism_dataset
from mechanism.factorial import CANONICAL_NAMED, build_selector

BASELINE_COLUMNS: list[str] = [
    "method",
    "source",
    "dataset",
    "k",
    "seed",
    "recovery_f1",
    "noise_rate",
    "runtime_s",
]


def _mrmr_selection_classif(
    X_train: pd.DataFrame, y_train: pd.Series, k: int, task: str, seed: int
) -> list[int]:
    """Thin adapter over ``benchmarks.cells.MRMRSelectionBaseline`` (smazzanti mrmr).

    ``MRMRSelectionBaseline.select`` already picks ``mrmr_classif``/``mrmr_regression``
    by ``task`` and maps selected column names -> integer indices; ``seed`` is unused
    (the package's own algorithm is deterministic given data, per the task brief).
    """
    del seed  # deterministic given data; no seed to thread through the external package
    return MRMRSelectionBaseline(name="mrmr_selection_classif").select(X_train, y_train, task, k)


def _skfeature_adapter(
    baseline_name: str,
) -> Callable[[pd.DataFrame, pd.Series, int, str, int], list[int]]:
    """Wrap a ``benchmarks.cells.BASELINES`` skfeature adapter as an EXTERNAL_METHODS entry.

    ``benchmarks.cells.SkfeatureBaseline`` self-guards on ``importlib.util.find_spec
    ("skfeature")`` and raises ``RuntimeError`` if the optional dependency is absent --
    that guard is reused as-is, not duplicated here.
    """

    def _select(
        X_train: pd.DataFrame, y_train: pd.Series, k: int, task: str, seed: int
    ) -> list[int]:
        del seed  # skfeature's CMIM/JMI are deterministic given data
        return BASELINES[baseline_name].select(X_train, y_train, task, k)

    return _select


EXTERNAL_METHODS: dict[str, Callable[[pd.DataFrame, pd.Series, int, str, int], list[int]]] = {
    "mrmr_selection_classif": _mrmr_selection_classif,
    "cmim": _skfeature_adapter("cmim"),
    "jmi": _skfeature_adapter("jmi"),
}

# CMIM/JMI (via benchmarks.cells.SkfeatureBaseline) are classification-only and raise
# ValueError on a regression task; the golden set mixes classification and regression
# datasets, so a (method, dataset) cell like (cmim, parabola) is infeasible by
# construction -- skipped and reported, mirroring the k > n_features skip below, not a
# silently-dropped unknown method name.
_CLASSIFICATION_ONLY_METHODS = frozenset({"cmim", "jmi"})


def _run_one(
    method: str, X_train: pd.DataFrame, y_train: pd.Series, task: str, k: int, seed: int
) -> list[int]:
    """Dispatch ``method`` to an external adapter or a modmrmr ``CANONICAL_NAMED`` spec.

    Raises ``ValueError`` listing valid names if ``method`` is in neither registry.
    """
    if method in EXTERNAL_METHODS:
        return EXTERNAL_METHODS[method](X_train, y_train, k, task, seed)
    if method in CANONICAL_NAMED:
        selector = build_selector(CANONICAL_NAMED[method], task, k, None, seed)
        selector.fit(X_train, y_train)
        return list(selector.selected_idx_)
    valid = sorted(EXTERNAL_METHODS) + sorted(CANONICAL_NAMED)
    raise ValueError(f"Unknown method {method!r}; valid methods: {valid}")


def _source_of(method: str) -> str:
    return "external" if method in EXTERNAL_METHODS else "modmrmr"


def run_baseline_grid(
    methods: Sequence[str],
    datasets: Sequence[str],
    ks: Sequence[int],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Run every (dataset x seed x method x k) cell; grade recovery against ground truth.

    Per (dataset, seed): loads the golden dataset, splits 70/30 leakage-free
    (``train_test_split(..., random_state=seed, stratify=y if task=="classification"
    else None)``), fits/selects each method x k on the train split only, and scores
    :func:`analysis.highdim_study.recovery_metrics` against ``gt``. Prints one
    flushed progress line per (dataset, seed) shard. Returns a DataFrame with
    columns == :data:`BASELINE_COLUMNS` (empty frame with those columns if no rows).
    """
    unknown = [m for m in methods if m not in EXTERNAL_METHODS and m not in CANONICAL_NAMED]
    if unknown:
        valid = sorted(EXTERNAL_METHODS) + sorted(CANONICAL_NAMED)
        raise ValueError(f"Unknown method(s) {unknown}; valid methods: {valid}")

    rows: list[dict] = []
    total = len(datasets) * len(seeds)
    done = 0
    for dataset in datasets:
        X, y, task, gt = load_mechanism_dataset(dataset)
        n_features = X.shape[1]
        for seed in seeds:
            X_train, _X_test, y_train, _y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=y if task == "classification" else None,
            )
            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)

            shard_rows = 0
            for method in methods:
                if method in _CLASSIFICATION_ONLY_METHODS and task != "classification":
                    print(
                        f"skipping {method!r} on {dataset!r}: classification-only, task={task!r}",
                        flush=True,
                    )
                    continue
                for k in ks:
                    if k > n_features:
                        continue
                    t0 = time.perf_counter()
                    selected = _run_one(method, X_train, y_train, task, k, seed)
                    runtime_s = time.perf_counter() - t0
                    metrics = recovery_metrics(selected, gt)
                    rows.append(
                        {
                            "method": method,
                            "source": _source_of(method),
                            "dataset": dataset,
                            "k": k,
                            "seed": seed,
                            "recovery_f1": metrics["recovery_f1"],
                            "noise_rate": metrics["noise_rate"],
                            "runtime_s": runtime_s,
                        }
                    )
                    shard_rows += 1

            done += 1
            print(
                f"[{done}/{total}] {dataset} seed={seed} rows={shard_rows}",
                flush=True,
            )

    if not rows:
        return pd.DataFrame(columns=BASELINE_COLUMNS)
    return pd.DataFrame(rows)[BASELINE_COLUMNS]
