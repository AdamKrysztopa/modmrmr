"""Resumable high-dimensional operator study: synthetic p-sweep + real OpenML sets.

Supersedes analysis/highdim_operator_study.py (run-at-import script, p fixed at 500).
Parameterizes p (feature count), sweeps the four operator specs, checkpoints each
(dataset, seed) shard to disk (resumable), and injects the vectorized
``fast_pearson_penalty`` redundancy path so p up to 10000 stays tractable.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from benchmarks.datasets import load_dataset
from mechanism.factorial import SelectorSpec, build_selector
from mechanism.ground_truth import GroundTruth
from mechanism.protocol import _downstream
from mechanism.recovery import ranking_scores, recovery
from modmrmr.core.scorers.base import fast_pearson_penalty

HIGHDIM_COLUMNS = [
    "kind",
    "dataset",
    "p",
    "operator",
    "aggregation",
    "k",
    "seed",
    "recovery_f1",
    "noise_rate",
    "recall_informative",
    "average_precision",
    "roc_auc",
    "downstream_score",
    "runtime_s",
]

SPECS = {
    "difference": SelectorSpec("pearson", "pearson_abs", "difference", "mean"),
    "multiplicative": SelectorSpec("pearson", "pearson_abs", "multiplicative", "mean"),
    "quotient": SelectorSpec("pearson", "pearson_abs", "quotient", "mean"),
    "mult_max": SelectorSpec("pearson", "pearson_abs", "multiplicative", "max"),
}

REAL_DATASETS = ["madelon", "isolet", "riboflavin", "gisette", "arcene"]


def make_highdim_synthetic(
    p: int, seed: int, n: int = 2600
) -> tuple[pd.DataFrame, pd.Series, GroundTruth]:
    """Madelon-style synthetic set: 5 informative + 15 redundant + (p - 20) noise."""
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=5,
        n_redundant=15,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=1.0,
        shuffle=False,
        random_state=seed,
    )
    gt = GroundTruth(
        informative=tuple(range(5)),
        codependent=(tuple(range(5, 20)),),
        noise=tuple(range(20, p)),
        dependence="mixed",
        n_features=p,
    )
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target"), gt


def recovery_metrics(selected: Sequence[int], gt: GroundTruth) -> dict[str, float]:
    """Recovery F1 (precision/recall over ``gt.relevant_columns``) + noise rate."""
    sel = set(selected)
    relevant = set(gt.relevant_columns)
    noise = set(gt.noise)
    tp = len(sel & relevant)
    precision = tp / len(sel) if sel else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recovery_f1": f1, "noise_rate": (len(sel & noise) / len(sel)) if sel else 0.0}


def _load_checkpoint(path: Path) -> pd.DataFrame | None:
    """Load a shard checkpoint if present, else ``None`` (never fatal on a bad file)."""
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except Exception as exc:  # noqa: BLE001 - corrupt checkpoint recomputed, not fatal
                print(
                    f"could not read checkpoint {csv_path} ({type(exc).__name__}: {exc}); "
                    "recomputing",
                    flush=True,
                )
                return None
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint is recomputed, not fatal
        print(f"could not read checkpoint {path} ({type(exc).__name__}: {exc}); recomputing")
        return None


def _write_checkpoint(df: pd.DataFrame, path: Path) -> None:
    """Write a shard checkpoint, tolerating a missing parquet engine (csv sidecar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001 - no parquet engine; fall back to csv checkpoint
        csv_path = path.with_suffix(".csv")
        print(f"no parquet engine for checkpoint ({type(exc).__name__}: {exc}); csv at {csv_path}")
        df.to_csv(csv_path, index=False)


def _fit_selector(spec: SelectorSpec, task: str, k: int, seed: int, X_train, y_train):
    """Build a selector for ``spec``, inject the fast pearson path, time the fit."""
    selector = build_selector(spec, task, k, None, seed)
    selector.redundancy = fast_pearson_penalty
    t0 = time.perf_counter()
    selector.fit(X_train, y_train)
    runtime_s = time.perf_counter() - t0
    return selector, runtime_s


def _synthetic_shard(
    p: int, seed: int, ks: list[int], checkpoint_dir: Path, done: int, total: int
) -> pd.DataFrame:
    label = f"synthetic_p{p}"
    path = checkpoint_dir / f"{label}__{seed}.parquet"
    t0 = time.perf_counter()
    cached = _load_checkpoint(path)
    if cached is not None:
        print(
            f"[{done}/{total}] {label} seed={seed} RESUMED from checkpoint "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return cached

    X, y, gt = make_highdim_synthetic(p=p, seed=seed)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    rows: list[dict] = []
    for _name, spec in SPECS.items():
        for k in ks:
            selector, runtime_s = _fit_selector(spec, "classification", k, seed, X_train, y_train)
            order = list(selector.selected_idx_)  # already in greedy pick-order
            metrics = recovery_metrics(order, gt)
            # Rank-based average precision over the selection order (positive class =
            # gt.informative). Unlike fixed-k F1 this is not recall-ceiling'd at k, so it
            # is the headline high-dim metric: it measures whether the operator ranks the
            # true signal above noise as dimension grows.
            rank = ranking_scores(order, gt)
            # recall over the true-informative set at k: of the informative features, how
            # many did the operator surface in its top-k pick? (coverage companion to AP).
            recall_informative = recovery(order, gt).recall
            rows.append(
                {
                    "kind": "synthetic",
                    "dataset": label,
                    "p": p,
                    "operator": spec.operator,
                    "aggregation": spec.aggregation,
                    "k": k,
                    "seed": seed,
                    "recovery_f1": metrics["recovery_f1"],
                    "noise_rate": metrics["noise_rate"],
                    "recall_informative": recall_informative,
                    "average_precision": rank.average_precision,
                    "roc_auc": rank.roc_auc,
                    "downstream_score": float("nan"),
                    "runtime_s": runtime_s,
                }
            )

    df = pd.DataFrame(rows, columns=HIGHDIM_COLUMNS)
    _write_checkpoint(df, path)
    print(
        f"[{done}/{total}] {label} seed={seed} rows={len(df)} {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return df


def run_synthetic_sweep(
    ps: list[int], seeds: list[int], ks: list[int], *, checkpoint_dir: Path | str
) -> pd.DataFrame:
    """Run the synthetic p-sweep across ``ps x seeds``, checkpointed per (p, seed) shard."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shards = [(p, seed) for p in ps for seed in seeds]
    total = len(shards)
    print(f"running {total} synthetic shards ({len(ps)} p-values x {len(seeds)} seeds)", flush=True)
    frames = [
        _synthetic_shard(p, seed, ks, checkpoint_dir, i, total)
        for i, (p, seed) in enumerate(shards, start=1)
    ]
    if not frames:
        return pd.DataFrame(columns=HIGHDIM_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _real_shard(name: str, seed: int, ks: list[int], checkpoint_dir: Path, done: int, total: int):
    path = checkpoint_dir / f"{name}__{seed}.parquet"
    t0 = time.perf_counter()
    cached = _load_checkpoint(path)
    if cached is not None:
        print(
            f"[{done}/{total}] {name} seed={seed} RESUMED from checkpoint "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return cached

    try:
        X, y, task = load_dataset(name)
    except Exception as exc:  # noqa: BLE001 - a single bad/network dataset never aborts the sweep
        print(
            f"[{done}/{total}] SKIP {name} seed={seed}: {type(exc).__name__}: {str(exc)[:160]}",
            flush=True,
        )
        return pd.DataFrame(columns=HIGHDIM_COLUMNS)

    is_clf = task == "classification"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y if is_clf else None
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    p = X.shape[1]

    rows: list[dict] = []
    for _spec_name, spec in SPECS.items():
        for k in ks:
            selector, runtime_s = _fit_selector(spec, task, k, seed, X_train, y_train)
            idx = list(selector.selected_idx_)
            downstream_score = _downstream(X_train, X_test, y_train, y_test, idx, task)
            rows.append(
                {
                    "kind": "real",
                    "dataset": name,
                    "p": p,
                    "operator": spec.operator,
                    "aggregation": spec.aggregation,
                    "k": k,
                    "seed": seed,
                    "recovery_f1": float("nan"),
                    "noise_rate": float("nan"),
                    "recall_informative": float("nan"),  # no ground truth for real sets
                    "average_precision": float("nan"),
                    "roc_auc": float("nan"),
                    "downstream_score": downstream_score,
                    "runtime_s": runtime_s,
                }
            )

    df = pd.DataFrame(rows, columns=HIGHDIM_COLUMNS)
    _write_checkpoint(df, path)
    print(
        f"[{done}/{total}] {name} seed={seed} rows={len(df)} {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return df


def run_real_benchmarks(
    names: list[str], seeds: list[int], ks: list[int], *, checkpoint_dir: Path | str
) -> pd.DataFrame:
    """Run the real-OpenML downstream sweep across ``names x seeds``, checkpointed per shard."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shards = [(name, seed) for name in names for seed in seeds]
    total = len(shards)
    print(
        f"running {total} real-dataset shards ({len(names)} datasets x {len(seeds)} seeds)",
        flush=True,
    )
    frames = [
        _real_shard(name, seed, ks, checkpoint_dir, i, total)
        for i, (name, seed) in enumerate(shards, start=1)
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=HIGHDIM_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.highdim_study", description=__doc__)
    parser.add_argument("--ps", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--real",
        nargs="+",
        default=REAL_DATASETS,
        help=f"real datasets to sweep, or 'none' to skip; default {REAL_DATASETS}",
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true", help="skip the synthetic p-sweep entirely"
    )
    parser.add_argument("--out", type=str, default="results/highdim_study.parquet")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing shard checkpoints and recompute every shard",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    out = Path(args.out)
    checkpoint_dir = out.parent / "_highdim_shards"
    if args.fresh and checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*"):
            f.unlink()

    frames: list[pd.DataFrame] = []

    if not args.skip_synthetic:
        synthetic_df = run_synthetic_sweep(
            args.ps, args.seeds, args.ks, checkpoint_dir=checkpoint_dir
        )
        if not synthetic_df.empty:
            frames.append(synthetic_df)
    else:
        print("skipping synthetic sweep (--skip-synthetic)", flush=True)

    real_names = [] if args.real == ["none"] else list(args.real)
    if real_names:
        real_df = run_real_benchmarks(
            real_names, args.seeds, args.ks, checkpoint_dir=checkpoint_dir
        )
        if not real_df.empty:
            frames.append(real_df)
    else:
        print("skipping real benchmarks (--real none)", flush=True)

    if not frames:
        print("highdim study produced no rows (check --ps/--real/--skip-synthetic combination)")
        return 1

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(HIGHDIM_COLUMNS[:7], kind="stable").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
        out_written = out
    except Exception as exc:  # noqa: BLE001 - no parquet engine; fall back to csv
        csv_path = out.with_suffix(".csv")
        print(f"no parquet engine for output ({type(exc).__name__}: {exc}); csv at {csv_path}")
        df.to_csv(csv_path, index=False)
        out_written = csv_path

    summary_path = out.with_name("highdim_study_summary.csv")
    summary = df.groupby(["kind", "dataset", "p", "operator", "aggregation", "k"], as_index=False)[
        [
            "recovery_f1",
            "noise_rate",
            "recall_informative",
            "average_precision",
            "roc_auc",
            "downstream_score",
            "runtime_s",
        ]
    ].mean()
    summary.to_csv(summary_path, index=False)

    print(
        f"wrote {len(df)} rows -> {out_written} "
        f"({df['dataset'].nunique()} datasets x {df['operator'].nunique()} operators); "
        f"summary -> {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
