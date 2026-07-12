"""CLI entry point: the fast, verbose, resumable full-factorial mechanism study.

Same study as :mod:`mechanism.run_factorial` (and same ``FACTORIAL_COLUMNS``
outputs -- results parquet, golden ``_summary.csv``, ``_benchmark_summary.csv``,
``_decision_guide.csv``, and the design-space figures), but:

* Each ``(dataset, seed)`` shard is driven by the memoized fast grid
  (:mod:`mechanism.fast_factorial_protocol`) instead of the reference oracle --
  numerically identical rows (proved in ``tests/mechanism/test_fast_equivalence.py``),
  computed once per measure-pair instead of once per ``(spec, k, threshold)``.
* **Verbose + checkpointed** (the "heavy scripts must be verbose + checkpointed"
  rule): every shard writes its own result to
  ``results/_shards/<dataset>__<seed>.parquet`` the instant it completes and
  prints a flushed ``[done/total] dataset seed=.. rows=.. elapsed`` progress line,
  and ``joblib.Parallel(verbose=10)`` reports scheduling.
* **Resumable**: on startup an already-present shard checkpoint is loaded and its
  computation skipped, so a killed run resumes where it left off. ``--fresh``
  ignores (and overwrites) existing checkpoints.

Reuses ``mechanism.run_factorial``'s spec/dataset resolution, result-writing, and
figure helpers so the final-output stage is identical to the committed CLI.

Example:

    uv run python -m mechanism.run_factorial_fast \\
        --datasets parabola radial sine \\
        --benchmark-datasets breast_cancer synthetic_clf diabetes friedman1 \\
        --specs all --ks 1 2 3 5 8 10 --seeds 0 1 2 --out results/factorial.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from benchmarks.datasets import list_datasets, load_dataset
from mechanism.datasets import list_mechanism_datasets
from mechanism.factorial import CANONICAL_NAMED, FULL_FACTORIAL, SelectorSpec
from mechanism.factorial_protocol import FACTORIAL_COLUMNS
from mechanism.fast_factorial_protocol import (
    run_fast_downstream_only_grid,
    run_fast_factorial_grid,
)
from mechanism.figures import (
    decision_guide_table,
    factorial_summary,
    features_vs_threshold,
    operator_aggregation_heatmap,
    ranking_leaderboard,
    recovery_vs_k,
)
from mechanism.run_factorial import (
    _DEFAULT_BENCHMARK_DATASETS,
    _SORT_KEYS,
    _resolve_benchmark_datasets,
    _resolve_golden_datasets,
    _resolve_specs,
    _write_results,
)


def _shard_checkpoint_path(shards_dir: Path, dataset: str, seed: int) -> Path:
    return shards_dir / f"{dataset}__{seed}.parquet"


def _load_checkpoint(path: Path) -> pd.DataFrame | None:
    """Load a shard checkpoint if present, else ``None`` (never fatal on a bad file)."""
    if not path.exists():
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


def _run_shard(
    kind: str,
    specs: list[SelectorSpec],
    dataset: str,
    ks: list[int],
    thresholds: list[float] | None,
    seed: int,
    shards_dir: Path,
    done: int,
    total: int,
    *,
    fresh: bool,
) -> pd.DataFrame:
    """Compute (or resume) one ``(dataset, seed)`` shard, checkpoint it, print progress.

    ``kind`` is ``"golden"`` (fast recovery+downstream grid) or ``"benchmark"``
    (fast downstream-only grid). If ``fresh`` is False and a checkpoint parquet
    already exists it is loaded and returned without recomputation.
    """
    path = _shard_checkpoint_path(shards_dir, dataset, seed)
    t0 = time.perf_counter()
    if not fresh:
        cached = _load_checkpoint(path)
        if cached is not None:
            print(
                f"[{done}/{total}] {dataset} seed={seed} rows={len(cached)} "
                f"RESUMED from checkpoint {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            return cached

    if kind == "golden":
        df = run_fast_factorial_grid(
            specs, [dataset], ks, thresholds, [seed], include_downstream=True
        )
    else:
        df = run_fast_downstream_only_grid(specs, [dataset], ks, thresholds, [seed])

    _write_checkpoint(df, path)
    elapsed = time.perf_counter() - t0
    print(
        f"[{done}/{total}] {dataset} seed={seed} rows={len(df)} {elapsed:.1f}s",
        flush=True,
    )
    return df


def _run_parallel_shards(
    kind: str,
    specs: list[SelectorSpec],
    datasets: list[str],
    ks: list[int],
    thresholds: list[float] | None,
    seeds: list[int],
    shards_dir: Path,
    jobs: int,
    counter_start: int,
    total: int,
    *,
    fresh: bool,
) -> list[pd.DataFrame]:
    """Run every ``(dataset, seed)`` shard of ``kind`` in parallel with per-shard checkpoints."""
    shards = [(dataset, seed) for dataset in datasets for seed in seeds]
    if not shards:
        return []
    return Parallel(n_jobs=jobs, verbose=10)(
        delayed(_run_shard)(
            kind,
            specs,
            dataset,
            ks,
            thresholds,
            seed,
            shards_dir,
            counter_start + i + 1,
            total,
            fresh=fresh,
        )
        for i, (dataset, seed) in enumerate(shards)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechanism.run_factorial_fast", description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"golden-set datasets, or 'none' to skip; one of {list_mechanism_datasets()}",
    )
    parser.add_argument(
        "--benchmark-datasets",
        nargs="+",
        default=list(_DEFAULT_BENCHMARK_DATASETS),
        help="benchmark datasets (downstream-only); 'all' attempts every registered set, "
        "'none' skips benchmark entirely",
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        default=None,
        help="SelectorSpec labels or CANONICAL_NAMED keys (e.g. 'ModMRMR'); 'all' for the "
        "full factorial grid",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 10])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=str, default="results/factorial.parquet")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--jobs", type=int, default=-1, help="joblib n_jobs; -1 = all cores")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing shard checkpoints and recompute every shard",
    )
    parser.add_argument(
        "--list", action="store_true", help="print available specs/datasets and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("specs:", [spec.label for spec in FULL_FACTORIAL])
        print("canonical named specs:", list(CANONICAL_NAMED))
        print("golden datasets:", list_mechanism_datasets())
        print("benchmark datasets:", list_datasets())
        return 0

    specs, unknown_specs = _resolve_specs(args.specs)
    golden_datasets, unknown_datasets = _resolve_golden_datasets(args.datasets)
    if unknown_specs or unknown_datasets:
        problems = []
        if unknown_specs:
            problems.append(f"unknown specs {unknown_specs}")
        if unknown_datasets:
            problems.append(
                f"unknown datasets {unknown_datasets}; known: {sorted(list_mechanism_datasets())}"
            )
        parser.error("; ".join(problems))

    benchmark_requested = _resolve_benchmark_datasets(args.benchmark_datasets)
    benchmark_datasets: list[str] = []
    skipped_benchmarks: list[tuple[str, str]] = []
    for name in benchmark_requested:
        try:
            load_dataset(name)
        except Exception as exc:  # noqa: BLE001 - a single bad/network dataset never aborts
            reason = f"{type(exc).__name__}: {exc}"
            skipped_benchmarks.append((name, reason))
            print(f"skipping benchmark dataset {name!r}: {reason}")
        else:
            benchmark_datasets.append(name)

    out = Path(args.out)
    shards_dir = out.parent / "_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    n_golden = len(golden_datasets) * len(args.seeds)
    n_benchmark = len(benchmark_datasets) * len(args.seeds)
    total = n_golden + n_benchmark
    print(
        f"running {total} shards ({n_golden} golden + {n_benchmark} benchmark) "
        f"x {len(specs)} specs; checkpoints -> {shards_dir}"
        f"{' (--fresh: ignoring existing checkpoints)' if args.fresh else ''}",
        flush=True,
    )

    golden_frames = _run_parallel_shards(
        "golden",
        specs,
        golden_datasets,
        args.ks,
        args.thresholds,
        args.seeds,
        shards_dir,
        args.jobs,
        0,
        total,
        fresh=args.fresh,
    )
    benchmark_frames = _run_parallel_shards(
        "benchmark",
        specs,
        benchmark_datasets,
        args.ks,
        args.thresholds,
        args.seeds,
        shards_dir,
        args.jobs,
        n_golden,
        total,
        fresh=args.fresh,
    )

    frames = golden_frames + benchmark_frames
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(_SORT_KEYS, kind="stable").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=FACTORIAL_COLUMNS)

    if df.empty:
        print(
            "factorial grid produced no rows "
            "(check --datasets/--benchmark-datasets/--specs/--ks combination)"
        )
        return 1

    out = _write_results(df, out)

    # factorial_summary means downstream_score per (spec, dependence). Benchmark rows
    # share dependence labels with golden rows but come from a different population, so
    # restrict the main summary to recovery-graded (golden) rows -- precision is non-NaN
    # only when a GroundTruth was available.
    graded = df.dropna(subset=["precision"])
    summary_path = out.with_name(f"{out.stem}_summary.csv")
    factorial_summary(graded).to_csv(summary_path, index=False)

    benchmark_rows = df[df["precision"].isna() & df["downstream_score"].notna()]
    benchmark_summary_path: Path | None = None
    if not benchmark_rows.empty:
        benchmark_summary_path = out.with_name(f"{out.stem}_benchmark_summary.csv")
        benchmark_rows.groupby(["spec", "dataset", "stop_mode"], as_index=False)[
            "downstream_score"
        ].mean().to_csv(benchmark_summary_path, index=False)

    decision_guide_path = out.with_name(f"{out.stem}_decision_guide.csv")
    decision_guide_table(df.dropna(subset=["precision"])).to_csv(decision_guide_path, index=False)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written_figures: list[Path] = []

    def _safe_figure(label: str, fn, *fn_args) -> None:
        try:
            written_figures.append(fn(*fn_args))
        except Exception as exc:  # noqa: BLE001 - one bad figure never kills the run
            print(f"figure {label!r} failed: {type(exc).__name__}: {exc}")

    _safe_figure(
        "operator_aggregation_heatmap[f1]",
        operator_aggregation_heatmap,
        df,
        "f1",
        figures_dir / "operator_aggregation_heatmap_f1.png",
    )
    _safe_figure(
        "operator_aggregation_heatmap[noise_rate]",
        operator_aggregation_heatmap,
        df,
        "noise_rate",
        figures_dir / "operator_aggregation_heatmap_noise_rate.png",
    )
    _safe_figure(
        "ranking_leaderboard", ranking_leaderboard, df, figures_dir / "ranking_leaderboard.png"
    )
    for dataset in golden_datasets:
        _safe_figure(
            f"features_vs_threshold[{dataset}]",
            features_vs_threshold,
            df,
            dataset,
            figures_dir / f"features_vs_threshold_{dataset}.png",
        )
        _safe_figure(
            f"recovery_vs_k[{dataset}]",
            recovery_vs_k,
            df,
            dataset,
            figures_dir / f"recovery_vs_k_{dataset}.png",
        )

    print(
        f"wrote {len(df)} rows -> {out} "
        f"({df['dataset'].nunique()} datasets x {df['spec'].nunique()} specs); "
        f"summary -> {summary_path}; decision guide -> {decision_guide_path}; "
        f"{len(written_figures)} figures -> {figures_dir}"
    )
    if benchmark_summary_path is not None:
        print(f"benchmark downstream summary -> {benchmark_summary_path}")
    if skipped_benchmarks:
        print(f"skipped {len(skipped_benchmarks)} benchmark datasets: {skipped_benchmarks}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
