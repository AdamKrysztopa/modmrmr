"""CLI entry point: external-baseline comparison on the golden sets (Task 7 / H5).

Runs external reference mRMR implementations (mrmr-selection, skfeature
CMIM/JMI) alongside our canonical named modmrmr specs
(:data:`mechanism.factorial.CANONICAL_NAMED`) on the mechanism-suite golden
datasets, under the same leakage-free train/test split used elsewhere in the
mechanism suite, and writes one tidy CSV
(:data:`mechanism.baseline_comparison.BASELINE_COLUMNS`). This is what lets the
paper's design-space claims be checked against something other than modmrmr
itself.

Example:

    uv run python -m mechanism.run_baseline_comparison \\
        --methods mrmr_selection_classif cmim jmi MID MIQ FCD FCQ ModMRMR \\
        --ks 3 5 10 --seeds 0 1 2 --out results/baseline_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mechanism.baseline_comparison import (
    EXTERNAL_METHODS,
    run_baseline_grid,
)
from mechanism.datasets import list_mechanism_datasets
from mechanism.factorial import CANONICAL_NAMED

_DEFAULT_METHODS: list[str] = list(EXTERNAL_METHODS) + list(CANONICAL_NAMED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechanism.run_baseline_comparison", description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=_DEFAULT_METHODS,
        help=f"method names; one of {_DEFAULT_METHODS}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_mechanism_datasets(),
        help=f"golden-set datasets; one of {list_mechanism_datasets()}",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=str, default="results/baseline_comparison.csv")
    return parser


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recovery_f1/noise_rate per method, for the printed run summary."""
    return (
        df.groupby("method")[["recovery_f1", "noise_rate"]]
        .mean()
        .rename(columns={"recovery_f1": "mean_recovery_f1", "noise_rate": "mean_noise_rate"})
        .reset_index()
        .sort_values("mean_recovery_f1", ascending=False, kind="stable")
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    known_datasets = set(list_mechanism_datasets())
    unknown_datasets = [d for d in args.datasets if d not in known_datasets]
    if unknown_datasets:
        parser.error(f"unknown datasets {unknown_datasets}; known: {sorted(known_datasets)}")

    known_methods = set(EXTERNAL_METHODS) | set(CANONICAL_NAMED)
    unknown_methods = [m for m in args.methods if m not in known_methods]
    if unknown_methods:
        parser.error(f"unknown methods {unknown_methods}; known: {sorted(known_methods)}")

    total = len(args.datasets) * len(args.seeds)
    print(
        f"running {total} shards ({len(args.datasets)} datasets x {len(args.seeds)} seeds) "
        f"x {len(args.methods)} methods x {len(args.ks)} ks",
        flush=True,
    )

    df = run_baseline_grid(args.methods, args.datasets, args.ks, args.seeds)

    if df.empty:
        print(
            "baseline-comparison grid produced no rows "
            "(check --methods/--datasets/--ks/--seeds combination)"
        )
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = _summary_table(df)
    print(
        f"wrote {len(df)} rows -> {out} "
        f"({df['dataset'].nunique()} datasets x {df['method'].nunique()} methods)"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
