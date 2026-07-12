"""CLI entry point: the MI-estimator comparison study (Study 4).

Holds redundancy (``pearson_abs``), operator (``difference``), and aggregation
(``mean``) fixed and compares mutual-information estimators as the relevance
measure over the mechanism-suite golden datasets
(:func:`mechanism.mi_comparison.run_mi_comparison_grid`). Golden-only: recovery
over the golden ground-truth datasets is the primary, discriminating metric for
this study. Downstream-on-real-benchmark-dataset comparison is explicitly OUT
OF SCOPE here -- it is a deferred secondary follow-up, not implemented in this
runner. Passing anything other than ``none`` to ``--benchmark-datasets`` prints
a notice and is skipped; it never crashes the run.

**Verbose + checkpointed** (the "heavy scripts must be verbose + checkpointed"
rule): each ``(dataset, seed)`` shard -- every estimator x every k for that
dataset/seed, computed via a single
:func:`mechanism.mi_comparison.run_mi_comparison_grid` call -- writes its own
result to ``results/_mi_shards/<dataset>__<seed>__<cfg>.parquet`` the instant it
completes and prints a flushed ``[done/total] dataset seed=.. rows=.. elapsed``
progress line; ``joblib.Parallel(verbose=10)`` reports scheduling. ``<cfg>`` is
an 8-hex-char hash of the sorted ``(estimators, ks)`` config (see
:func:`_config_hash`), so a shard from one ``--estimators``/``--ks`` combination
is never mistaken for -- or silently returned in place of -- a shard computed
for a different one.
**Resumable**: an already-present shard checkpoint is loaded and its
computation skipped, so a killed run resumes where it left off. ``--fresh``
ignores (and overwrites) existing checkpoints.

Example:

    uv run python -m mechanism.run_mi_comparison \\
        --estimators all --ks 1 2 3 5 8 10 --seeds 0 1 2 \\
        --out results/mi_comparison.parquet --figures-dir results/figures
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from mechanism.datasets import list_mechanism_datasets
from mechanism.figures import (
    mi_ap_by_dependence,
    mi_comparison_summary,
    mi_ranking_leaderboard,
    mi_winner,
)
from mechanism.mi_comparison import MI_COMPARISON_COLUMNS, MI_ESTIMATORS, run_mi_comparison_grid


def _resolve_golden_datasets(tokens: list[str] | None) -> tuple[list[str], list[str]]:
    """``None`` -> every golden dataset; ``["none"]`` -> none; else validate against registry."""
    if tokens is None:
        return list_mechanism_datasets(), []
    if tokens == ["none"]:
        return [], []
    known = set(list_mechanism_datasets())
    unknown = [name for name in tokens if name not in known]
    return [name for name in tokens if name in known], unknown


def _resolve_estimators(tokens: list[str] | None) -> tuple[list[str], list[str]]:
    """``None`` or ``["all"]`` -> every :data:`MI_ESTIMATORS`; else validate against it."""
    if tokens is None or tokens == ["all"]:
        return list(MI_ESTIMATORS), []
    known = set(MI_ESTIMATORS)
    unknown = [name for name in tokens if name not in known]
    return [name for name in tokens if name in known], unknown


def _resolve_benchmark_datasets(tokens: list[str]) -> list[str]:
    """Downstream-on-real-benchmark comparison is out of scope for this study.

    ``["none"]`` (the default) requests nothing. Anything else is reported and
    skipped -- never fatal, never silently dropped.
    """
    if tokens == ["none"]:
        return []
    print(
        f"benchmark-dataset downstream comparison is not implemented in "
        f"mechanism.run_mi_comparison (recovery over golden datasets is this study's "
        f"primary metric); skipping requested benchmark datasets: {tokens}",
        flush=True,
    )
    return []


def _config_hash(estimators: list[str], ks: list[int]) -> str:
    """Deterministic, order-independent cache key for an (estimators, ks) run config.

    Not a security hash -- sha1 truncated to 8 hex chars is plenty to keep shard
    checkpoints from different ``--estimators``/``--ks`` runs from colliding.
    """
    payload = repr((sorted(estimators), sorted(ks)))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def _shard_checkpoint_path(shards_dir: Path, dataset: str, seed: int, cfg: str) -> Path:
    # NOTE: assumes `dataset` names never contain "__" (true for the current
    # mechanism-suite registry); if that changes, this filename scheme needs
    # a real separator/sanitization pass.
    return shards_dir / f"{dataset}__{seed}__{cfg}.parquet"


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


def _run_shard(
    estimators: list[str],
    dataset: str,
    ks: list[int],
    seed: int,
    shards_dir: Path,
    done: int,
    total: int,
    cfg: str,
    *,
    fresh: bool,
) -> pd.DataFrame:
    """Compute (or resume) one ``(dataset, seed)`` shard, checkpoint it, print progress.

    A shard is every estimator x every k for one ``(dataset, seed)`` pair. If
    ``fresh`` is False and a checkpoint parquet already exists it is loaded and
    returned without recomputation. ``cfg`` (see :func:`_config_hash`) keys the
    checkpoint to the requested ``(estimators, ks)`` so a differently-scoped
    rerun never loads a stale, incompletely-scoped shard.
    """
    path = _shard_checkpoint_path(shards_dir, dataset, seed, cfg)
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

    df = run_mi_comparison_grid(estimators, [dataset], ks, [seed], include_downstream=True)

    _write_checkpoint(df, path)
    elapsed = time.perf_counter() - t0
    print(
        f"[{done}/{total}] {dataset} seed={seed} rows={len(df)} {elapsed:.1f}s",
        flush=True,
    )
    return df


def _run_parallel_shards(
    estimators: list[str],
    datasets: list[str],
    ks: list[int],
    seeds: list[int],
    shards_dir: Path,
    jobs: int,
    cfg: str,
    *,
    fresh: bool,
) -> list[pd.DataFrame]:
    """Run every ``(dataset, seed)`` shard in parallel with per-shard checkpoints."""
    shards = [(dataset, seed) for dataset in datasets for seed in seeds]
    if not shards:
        return []
    total = len(shards)
    return Parallel(n_jobs=jobs, verbose=10)(
        delayed(_run_shard)(
            estimators,
            dataset,
            ks,
            seed,
            shards_dir,
            i + 1,
            total,
            cfg,
            fresh=fresh,
        )
        for i, (dataset, seed) in enumerate(shards)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechanism.run_mi_comparison", description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"golden-set datasets, or 'none' to skip; one of {list_mechanism_datasets()}",
    )
    parser.add_argument(
        "--benchmark-datasets",
        nargs="+",
        default=["none"],
        help="OUT OF SCOPE for this study (downstream-on-real-benchmark is a deferred "
        "secondary metric); anything other than 'none' is reported and skipped, never fatal",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=list(MI_ESTIMATORS),
        help=f"MI estimator names, or 'all' for every one of {list(MI_ESTIMATORS)}",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=str, default="results/mi_comparison.parquet")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--jobs", type=int, default=-1, help="joblib n_jobs; -1 = all cores")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing shard checkpoints and recompute every shard",
    )
    parser.add_argument(
        "--list", action="store_true", help="print available estimators/datasets and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("estimators:", list(MI_ESTIMATORS))
        print("golden datasets:", list_mechanism_datasets())
        return 0

    golden_datasets, unknown_datasets = _resolve_golden_datasets(args.datasets)
    estimators, unknown_estimators = _resolve_estimators(args.estimators)
    if unknown_datasets or unknown_estimators:
        problems = []
        if unknown_datasets:
            problems.append(
                f"unknown datasets {unknown_datasets}; known: {sorted(list_mechanism_datasets())}"
            )
        if unknown_estimators:
            problems.append(
                f"unknown estimators {unknown_estimators}; known: {sorted(MI_ESTIMATORS)}"
            )
        parser.error("; ".join(problems))

    _resolve_benchmark_datasets(args.benchmark_datasets)

    out = Path(args.out)
    shards_dir = out.parent / "_mi_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config_hash(estimators, args.ks)

    total = len(golden_datasets) * len(args.seeds)
    print(
        f"running {total} shards ({len(golden_datasets)} datasets x {len(args.seeds)} seeds) "
        f"x {len(estimators)} estimators x {len(args.ks)} ks; checkpoints -> {shards_dir} "
        f"(cfg={cfg})"
        f"{' (--fresh: ignoring existing checkpoints)' if args.fresh else ''}",
        flush=True,
    )

    frames = _run_parallel_shards(
        estimators,
        golden_datasets,
        args.ks,
        args.seeds,
        shards_dir,
        args.jobs,
        cfg,
        fresh=args.fresh,
    )

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["dataset", "estimator", "k", "seed"], kind="stable").reset_index(
            drop=True
        )
    else:
        df = pd.DataFrame(columns=MI_COMPARISON_COLUMNS)

    if df.empty:
        print(
            "MI-comparison grid produced no rows "
            "(check --datasets/--estimators/--ks/--seeds combination)"
        )
        return 1

    out = _write_results(df, out)

    summary_path = out.with_name(f"{out.stem}_summary.csv")
    mi_comparison_summary(df).to_csv(summary_path, index=False)

    winner_table = mi_winner(df)
    winner_path = out.with_name(f"{out.stem}_winner.csv")
    winner_table.to_csv(winner_path, index=False)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written_figures: list[Path] = []

    def _safe_figure(label: str, fn, *fn_args) -> None:
        try:
            written_figures.append(fn(*fn_args))
        except Exception as exc:  # noqa: BLE001 - one bad figure never kills the run
            print(f"figure {label!r} failed: {type(exc).__name__}: {exc}")

    _safe_figure(
        "mi_ranking_leaderboard",
        mi_ranking_leaderboard,
        df,
        figures_dir / "mi_ranking_leaderboard.png",
    )
    _safe_figure(
        "mi_ap_by_dependence",
        mi_ap_by_dependence,
        df,
        figures_dir / "mi_ap_by_dependence.png",
    )

    non_regressing = winner_table[~winner_table["regresses_nonlinear"]]
    if not non_regressing.empty:
        winner_row = non_regressing.iloc[0]
        winner_summary = (
            f"winner={winner_row['estimator']} (mean_ap_golden={winner_row['mean_ap_golden']:.3f})"
        )
    else:
        winner_summary = "winner=<none, every estimator regresses on nonlinear dependence>"

    print(
        f"wrote {len(df)} rows -> {out} "
        f"({df['dataset'].nunique()} datasets x {df['estimator'].nunique()} estimators); "
        f"summary -> {summary_path}; winner -> {winner_path}; "
        f"{len(written_figures)} figures -> {figures_dir}; {winner_summary}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
