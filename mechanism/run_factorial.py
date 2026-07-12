"""CLI entry point: run the full-factorial mechanism-suite study, write tables + figures.

Ties the full 4-axis ``SelectorSpec`` grid (:mod:`mechanism.factorial`) to the golden-set
recovery+downstream runner (:func:`mechanism.factorial_protocol.run_factorial_grid`) and the
downstream-only benchmark runner (:func:`mechanism.factorial_protocol.run_downstream_only_grid`),
then regenerates the design-space figures and summary/decision-guide tables from the combined
result. Mirrors ``benchmarks.run``/``mechanism.run``'s CLI shape, including their ``--list``
fix: every argument is optional at the parser level and only validated when actually running the
grid, so ``--list`` works standalone.

Both runners are parallelized across ``(dataset, seed)`` shards via ``joblib.Parallel`` -- each
shard is an independent ``run_*_grid(specs, [dataset], ks, thresholds, [seed])`` call, so no
shared mutable state crosses shards and the seeded selection inside each shard is unaffected by
``--jobs``. Rows are sorted by a stable key after concatenation so the final table (and hence
downstream figures/tables) does not depend on shard completion order.

Example (deferred full run -- see task-D1: this module is validated on TINY configs only; the
full run is a later phase):

    uv run python -m mechanism.run_factorial \\
        --datasets parabola radial sine \\
        --benchmark-datasets breast_cancer synthetic_clf diabetes friedman1 \\
        --specs all --ks 1 2 3 5 8 10 --seeds 0 1 2 --out results/factorial.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from benchmarks.datasets import list_datasets, load_dataset
from mechanism.datasets import list_mechanism_datasets
from mechanism.factorial import CANONICAL_NAMED, FULL_FACTORIAL, SelectorSpec
from mechanism.factorial_protocol import (
    FACTORIAL_COLUMNS,
    run_downstream_only_grid,
    run_factorial_grid,
)
from mechanism.figures import (
    decision_guide_table,
    factorial_summary,
    features_vs_threshold,
    operator_aggregation_heatmap,
    ranking_leaderboard,
    recovery_vs_k,
)

_DEFAULT_BENCHMARK_DATASETS = ["breast_cancer", "synthetic_clf", "diabetes", "friedman1"]
_SORT_KEYS = ["dataset", "spec", "stop_mode", "k", "score_threshold", "seed"]


def _run_golden_shard(
    specs: list[SelectorSpec],
    dataset: str,
    ks: list[int],
    thresholds: list[float] | None,
    seed: int,
) -> pd.DataFrame:
    """One (dataset, seed) shard of the golden-set recovery+downstream grid."""
    return run_factorial_grid(specs, [dataset], ks, thresholds, [seed], include_downstream=True)


def _run_benchmark_shard(
    specs: list[SelectorSpec],
    dataset: str,
    ks: list[int],
    thresholds: list[float] | None,
    seed: int,
) -> pd.DataFrame:
    """One (dataset, seed) shard of the downstream-only benchmark grid."""
    return run_downstream_only_grid(specs, [dataset], ks, thresholds, [seed])


def _resolve_specs(tokens: list[str] | None) -> tuple[list[SelectorSpec], list[str]]:
    """Resolve ``--specs`` tokens to ``SelectorSpec``s -> ``(resolved, unknown_tokens)``.

    ``None`` or ``["all"]`` resolves to every :data:`FULL_FACTORIAL` spec. Each other token
    is looked up first as a :data:`CANONICAL_NAMED` key, then as a ``SelectorSpec.label``.
    """
    if tokens is None or tokens == ["all"]:
        return list(FULL_FACTORIAL), []
    label_lookup = {spec.label: spec for spec in FULL_FACTORIAL}
    resolved: list[SelectorSpec] = []
    unknown: list[str] = []
    for token in tokens:
        if token in CANONICAL_NAMED:
            resolved.append(CANONICAL_NAMED[token])
        elif token in label_lookup:
            resolved.append(label_lookup[token])
        else:
            unknown.append(token)
    return resolved, unknown


def _resolve_golden_datasets(tokens: list[str] | None) -> tuple[list[str], list[str]]:
    """``None`` -> every golden dataset; ``["none"]`` -> none; else validate against registry."""
    if tokens is None:
        return list_mechanism_datasets(), []
    if tokens == ["none"]:
        return [], []
    known = set(list_mechanism_datasets())
    unknown = [name for name in tokens if name not in known]
    return [name for name in tokens if name in known], unknown


def _resolve_benchmark_datasets(tokens: list[str]) -> list[str]:
    """``["none"]`` -> none requested; ``["all"]`` -> every registered benchmark set; else as-is."""
    if tokens == ["none"]:
        return []
    if tokens == ["all"]:
        return list_datasets()
    return tokens


def _run_parallel_grid(
    shard_fn, specs, datasets, ks, thresholds, seeds, jobs
) -> list[pd.DataFrame]:
    shards = [(dataset, seed) for dataset in datasets for seed in seeds]
    if not shards:
        return []
    return Parallel(n_jobs=jobs)(
        delayed(shard_fn)(specs, dataset, ks, thresholds, seed) for dataset, seed in shards
    )


def _write_results(df: pd.DataFrame, out_path: Path) -> Path:
    """Write ``df`` to ``out_path`` (parquet if ``.parquet``, else csv); falls back to csv."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception as exc:  # noqa: BLE001 - no parquet engine available; fall back
            csv_path = out_path.with_suffix(".csv")
            print(
                f"no parquet engine available ({type(exc).__name__}: {exc}); "
                f"falling back to csv at {csv_path}"
            )
            df.to_csv(csv_path, index=False)
            return csv_path
    df.to_csv(out_path, index=False)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechanism.run_factorial", description=__doc__)
    # All arguments are optional at the parser level so --list can be used standalone; the
    # rest are validated manually in main() only when an actual run is requested.
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

    golden_frames = _run_parallel_grid(
        _run_golden_shard, specs, golden_datasets, args.ks, args.thresholds, args.seeds, args.jobs
    )
    benchmark_frames = _run_parallel_grid(
        _run_benchmark_shard,
        specs,
        benchmark_datasets,
        args.ks,
        args.thresholds,
        args.seeds,
        args.jobs,
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

    out = _write_results(df, Path(args.out))

    # factorial_summary means downstream_score per (spec, dependence). Benchmark rows
    # share dependence labels with golden rows (both use linear/nonlinear/mixed) but come
    # from a different population, so pooling their downstream with golden downstream would
    # conflate two things into one paper number. Restrict the main summary to recovery-graded
    # (golden) rows -- precision is non-NaN only when a GroundTruth was available.
    graded = df.dropna(subset=["precision"])
    summary_path = out.with_name(f"{out.stem}_summary.csv")
    factorial_summary(graded).to_csv(summary_path, index=False)

    # Benchmark rows carry no recovery; report their real-world downstream separately
    # (per spec x dataset x stop_mode) so the signal is kept, not conflated above.
    benchmark_rows = df[df["precision"].isna() & df["downstream_score"].notna()]
    benchmark_summary_path: Path | None = None
    if not benchmark_rows.empty:
        benchmark_summary_path = out.with_name(f"{out.stem}_benchmark_summary.csv")
        benchmark_rows.groupby(["spec", "dataset", "stop_mode"], as_index=False)[
            "downstream_score"
        ].mean().to_csv(benchmark_summary_path, index=False)

    # decision_guide_table groups by dataset and calls idxmax() on mean f1 -- benchmark
    # rows (no GroundTruth) are always NaN f1, which idxmax() raises on for an all-NaN
    # group. Restrict to recovery-graded rows (precision is only non-NaN when a
    # GroundTruth was available) so a benchmark-only run degrades to an empty table
    # instead of crashing the CLI.
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
