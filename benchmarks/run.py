"""CLI entry point tying datasets x cells -> a results file.

Execution of the full published grid is a LATER phase. This module builds the wiring
and is validated on a tiny synthetic config. Example (deferred full run):

    uv run python -m benchmarks.run \
        --datasets breast_cancer diabetes \
        --cells MID MIQ FCD FCQ ModMRMR skb_f skb_mi mrmr_smazzanti relieff rfe \
        --ks 1 2 5 10 20 50 --cv 5 --seeds 0 1 2 --out results/grid.parquet
"""

from __future__ import annotations

import argparse

from benchmarks.cells import get_adapter, list_cells
from benchmarks.datasets import list_datasets
from benchmarks.protocol import run_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmarks.run", description=__doc__)
    # --datasets/--cells/--out are only required for an actual run; --list must work
    # without them, so they are optional here and validated manually in main().
    parser.add_argument("--datasets", nargs="+", help="dataset names")
    parser.add_argument("--cells", nargs="+", help=f"one of {list_cells()}")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--cv", type=int, default=5, help="CV fold count")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", help="output .parquet or .csv path")
    parser.add_argument(
        "--list", action="store_true", help="print available datasets/cells and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print("datasets:", list_datasets())
        print("cells:", list_cells())
        return 0

    missing = [name for name in ("datasets", "cells", "out") if getattr(args, name) is None]
    if missing:
        parser.error(
            "the following arguments are required: " + ", ".join(f"--{m}" for m in missing)
        )

    cells = [get_adapter(name) for name in args.cells]
    results = run_grid(
        cells=cells,
        datasets=args.datasets,
        learners=None,
        cv=args.cv,
        seeds=args.seeds,
        ks=args.ks,
    )

    out = args.out
    if out.endswith(".parquet"):
        results.to_parquet(out, index=False)
    else:
        results.to_csv(out, index=False)
    print(
        f"wrote {len(results)} rows -> {out} "
        f"({results['dataset'].nunique()} datasets x {results['method'].nunique()} methods)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
