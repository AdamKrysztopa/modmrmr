"""Sweep driver for the profiling benchmarks.

Streams one CSV row per measured cell and flushes immediately, so an
interruption at hour six costs one cell rather than the whole run. Re-running
the same command skips cells already present in the output.

Usage::

    uv run python -m benchmarks.profiling.run --benchmark scorer_scaling
    uv run python -m benchmarks.profiling.run --benchmark all --budget-s 120
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.profiling.data import make_matrix, make_pair
from benchmarks.profiling.grid import DEFAULT_GRIDS, ProfileCell, ProfileGrid
from benchmarks.profiling.timing import measure
from modmrmr.core.estimator import MRMRSelector
from modmrmr.core.scorers import as_penalty_matrix, get_scorer

CSV_COLUMNS = [
    "benchmark",
    "scorer",
    "n",
    "p",
    "data_kind",
    "seed",
    "status",
    "median_s",
    "iqr_s",
    "repeats",
    "projected_s",
]

DEFAULT_OUT_DIR = Path("results/profiling")


def project_seconds(observed: list[tuple[int, float]], target_n: int) -> float:
    """Project runtime at ``target_n`` from ``(n, seconds)`` observations.

    Fits ``log(t) = a + b*log(n)`` by least squares — i.e. assumes power-law
    scaling, which is what every kernel here exhibits. With a single
    observation there is no exponent to fit, so linear scaling is assumed;
    that under-projects a quadratic kernel, which is the safe direction (it
    risks running a cell rather than wrongly skipping it).

    Args:
        observed: ``(n, median_seconds)`` pairs already measured for this scorer.
        target_n: Sample size to project to.

    Returns:
        Projected wall-clock seconds. ``inf`` if no observations were given.
    """
    if not observed:
        return float("inf")
    if len(observed) == 1:
        n0, t0 = observed[0]
        return float(t0 * target_n / n0)
    ns = np.log(np.array([n for n, _ in observed], dtype=float))
    ts = np.log(np.array([max(t, 1e-12) for _, t in observed], dtype=float))
    slope, intercept = np.polyfit(ns, ts, 1)
    return float(np.exp(intercept + slope * np.log(target_n)))


def load_completed_keys(path: Path) -> set[tuple[str, str, int, int, str]]:
    """Return the identity keys of cells already recorded in ``path``."""
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int, int, str]] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            keys.add(
                (
                    row["benchmark"],
                    row["scorer"],
                    int(row["n"]),
                    int(row["p"]),
                    row["data_kind"],
                )
            )
    return keys


def _pair_workload(cell: ProfileCell):
    scorer = get_scorer(cell.scorer)
    x, y = make_pair(cell.n, cell.data_kind, seed=cell.seed)
    return lambda: scorer.score_pair(x, y, random_state=cell.seed)


def _driver_workload(cell: ProfileCell):
    scorer = get_scorer(cell.scorer)
    penalty = as_penalty_matrix(scorer, random_state=cell.seed)
    X, _ = make_matrix(cell.n, cell.p, cell.data_kind, seed=cell.seed)
    return lambda: penalty(X)


def _end_to_end_workload(cell: ProfileCell):
    X, y = make_matrix(cell.n, cell.p, cell.data_kind, seed=cell.seed)
    task = "classification" if cell.data_kind in ("discrete", "mixed") else "regression"

    def _fit() -> None:
        MRMRSelector(
            n_features=10,
            relevance=cell.scorer,
            redundancy=cell.scorer,
            task=task,
            random_state=cell.seed,
        ).fit(X, y)

    return _fit


_WORKLOADS = {
    "scorer_scaling": _pair_workload,
    "driver_scaling": _driver_workload,
    "end_to_end": _end_to_end_workload,
}


def run_cell(cell: ProfileCell, *, repeats: int, warmup: int) -> dict[str, object]:
    """Measure one cell and return its CSV row."""
    workload = _WORKLOADS[cell.benchmark](cell)
    timing = measure(workload, repeats=repeats, warmup=warmup)
    return {
        "benchmark": cell.benchmark,
        "scorer": cell.scorer,
        "n": cell.n,
        "p": cell.p,
        "data_kind": cell.data_kind,
        "seed": cell.seed,
        "status": "ok",
        "median_s": timing.median_s,
        "iqr_s": timing.iqr_s,
        "repeats": timing.repeats,
        "projected_s": "",
    }


def _skipped_row(cell: ProfileCell, projected: float, status: str) -> dict[str, object]:
    return {
        "benchmark": cell.benchmark,
        "scorer": cell.scorer,
        "n": cell.n,
        "p": cell.p,
        "data_kind": cell.data_kind,
        "seed": cell.seed,
        "status": status,
        "median_s": "",
        "iqr_s": "",
        "repeats": 0,
        "projected_s": projected,
    }


def load_observed(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Rebuild per-scorer ``(n, median_s)`` observations from prior ``ok`` rows.

    Seeds the cost-guard projection on a resumed run; without it the first
    remaining cell per scorer would run unguarded after an interruption.
    """
    observed: dict[str, list[tuple[int, float]]] = {}
    if not path.exists():
        return observed
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            observed.setdefault(row["scorer"], []).append((int(row["n"]), float(row["median_s"])))
    return observed


def run_grid(
    grid: ProfileGrid,
    out_path: Path,
    *,
    budget_s: float,
    repeats: int = 5,
    warmup: int = 1,
) -> None:
    """Run every cell in ``grid``, streaming rows to ``out_path``.

    Args:
        grid: Sweep specification.
        out_path: CSV to append to. Created with a header if absent.
        budget_s: Per-cell wall-clock budget. A cell whose projected cost
            exceeds this is skipped and recorded as ``skipped_projected_cost``.
        repeats: Timed runs per cell.
        warmup: Untimed runs per cell.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(out_path)
    cells = [c for c in grid.cells() if c.key not in completed]
    total = len(cells)
    if total == 0:
        print(f"[{grid.benchmark}] all cells already recorded in {out_path}", flush=True)
        return

    # Per-scorer observations feeding the cost-guard projection, seeded from
    # prior ok rows so the guard still works on a resumed run. Cells are run
    # in ascending n per scorer, so the cheap ones inform the expensive ones.
    observed = load_observed(out_path)
    write_header = not out_path.exists()
    start = time.perf_counter()

    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            fh.flush()

        for idx, cell in enumerate(sorted(cells, key=lambda c: (c.scorer, c.n, c.p)), start=1):
            seen = observed.setdefault(cell.scorer, [])
            # project_seconds extrapolates a per-repeat median; the budget is
            # per-cell wall-clock, so scale by the number of runs per cell.
            projected = project_seconds(seen, cell.n) * (repeats + warmup) if seen else 0.0
            if seen and projected > budget_s:
                row = _skipped_row(cell, projected, "skipped_projected_cost")
                print(
                    f"[{idx}/{total}] SKIP {cell.scorer} n={cell.n} p={cell.p} "
                    f"{cell.data_kind} — projected {projected:.1f}s > budget {budget_s:.1f}s",
                    flush=True,
                )
            else:
                try:
                    row = run_cell(cell, repeats=repeats, warmup=warmup)
                    seen.append((cell.n, float(row["median_s"])))
                except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
                    row = _skipped_row(cell, 0.0, "error")
                    row["projected_s"] = ""
                    print(
                        f"[{idx}/{total}] ERROR {cell.scorer} n={cell.n} "
                        f"p={cell.p} {cell.data_kind}: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    elapsed = time.perf_counter() - start
                    eta = elapsed / idx * (total - idx)
                    print(
                        f"[{idx}/{total}] {cell.scorer} n={cell.n} p={cell.p} "
                        f"{cell.data_kind} — {row['median_s']:.4f}s "
                        f"(IQR {row['iqr_s']:.4f}) ETA {eta / 60:.1f}min",
                        flush=True,
                    )
            writer.writerow(row)
            fh.flush()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the modmrmr profiling sweep.")
    parser.add_argument(
        "--benchmark",
        default="all",
        choices=[*DEFAULT_GRIDS, "all"],
        help="Which benchmark grid to run.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for CSV output."
    )
    parser.add_argument(
        "--budget-s",
        type=float,
        default=120.0,
        help="Per-cell wall-clock budget; cells projected above it are skipped.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Timed runs per cell.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed runs per cell.")
    args = parser.parse_args()

    names = list(DEFAULT_GRIDS) if args.benchmark == "all" else [args.benchmark]
    for name in names:
        out = args.out_dir / f"{name}.csv"
        print(f"=== {name} -> {out} ===", flush=True)
        run_grid(
            DEFAULT_GRIDS[name],
            out,
            budget_s=args.budget_s,
            repeats=args.repeats,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
