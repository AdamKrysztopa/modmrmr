"""CLI entry point: run the mechanism grid, write results + figures.

Ties the mechanism datasets (:mod:`mechanism.datasets`) and MRMR-family/baseline
cells (:mod:`benchmarks.cells`) together via :func:`mechanism.protocol.run_mechanism_grid`,
then regenerates the recovery figures and summary table from the result. Mirrors
``benchmarks.run``'s CLI shape, including its ``--list`` fix: ``--datasets``/``--cells``/
``--out`` are optional at the parser level and only required when actually running the
grid, so ``--list`` works standalone.

Example (deferred full run):

    uv run python -m mechanism.run \\
        --datasets parabola radial sine \\
        --cells MID MIQ MIFS ModMRMR ModMRMR_dcor skb_f skb_mi \\
        --ks 1 2 5 10 --seeds 0 1 2 --out results/mechanism.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.cells import list_cells
from mechanism.datasets import list_mechanism_datasets
from mechanism.figures import fixed_k_vs_threshold, linear_vs_nonlinear_gap, mechanism_summary
from mechanism.figures import recovery_vs_k as _recovery_vs_k
from mechanism.protocol import run_mechanism_grid

_DEFAULT_CELLS = [
    "MID",
    "MIQ",
    "MIFS",
    "ModMRMR",
    "ModMRMR_dcor",
    "skb_f",
    "skb_mi",
]


def _default_cells() -> list[str]:
    """The sensible default cell set, filtered down to whatever is registered."""
    available = set(list_cells())
    return [name for name in _DEFAULT_CELLS if name in available]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechanism.run", description=__doc__)
    # --datasets/--cells/--out are only required for an actual run; --list must work
    # without them, so they are optional here and validated manually in main().
    parser.add_argument(
        "--datasets", nargs="+", default=None, help=f"one of {list_mechanism_datasets()}"
    )
    parser.add_argument("--cells", nargs="+", default=None, help=f"one of {list_cells()}")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=None, help="score_threshold sweep values"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=str, default="results/mechanism.parquet")
    parser.add_argument(
        "--figures-dir", type=str, default="results/figures", help="where to write figures"
    )
    parser.add_argument(
        "--list", action="store_true", help="print available datasets/cells and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print("datasets:", list_mechanism_datasets())
        print("cells:", list_cells())
        return 0

    datasets = args.datasets if args.datasets is not None else list_mechanism_datasets()
    cells = args.cells if args.cells is not None else _default_cells()

    known_datasets = set(list_mechanism_datasets())
    known_cells = set(list_cells())
    unknown_datasets = [d for d in datasets if d not in known_datasets]
    unknown_cells = [c for c in cells if c not in known_cells]
    if unknown_datasets or unknown_cells:
        problems = []
        if unknown_datasets:
            problems.append(f"unknown datasets {unknown_datasets}; known: {sorted(known_datasets)}")
        if unknown_cells:
            problems.append(f"unknown cells {unknown_cells}; known: {sorted(known_cells)}")
        parser.error("; ".join(problems))

    df = run_mechanism_grid(
        cells=cells,
        datasets=datasets,
        ks=args.ks,
        seeds=args.seeds,
        thresholds=args.thresholds,
    )

    if df.empty:
        print("mechanism grid produced no rows (check --datasets/--cells/--ks combination)")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    summary = mechanism_summary(df)
    summary_path = out.with_name(f"{out.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written_figures = []
    for dataset in sorted(df["dataset"].unique()):
        fig_path = figures_dir / f"recovery_vs_k_{dataset}.png"
        written_figures.append(_recovery_vs_k(df, dataset, fig_path))
    written_figures.append(linear_vs_nonlinear_gap(df, figures_dir / "linear_vs_nonlinear_gap.png"))
    written_figures.append(fixed_k_vs_threshold(df, figures_dir / "fixed_k_vs_threshold.png"))

    print(
        f"wrote {len(df)} rows -> {out} "
        f"({df['dataset'].nunique()} datasets x {df['method'].nunique()} methods); "
        f"summary -> {summary_path}; {len(written_figures)} figures -> {figures_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
