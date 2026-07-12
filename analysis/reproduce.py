"""Regenerate every paper artifact (figures, tables, guidance) from a results
file. Runnable as ``python -m analysis.reproduce --results R.parquet --outdir D``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.figures import (
    auc_k_barchart,
    cd_diagram,
    grid_heatmap,
    stability_accuracy_scatter,
)
from analysis.guidance import build_decision_guide
from analysis.paths import ARTIFACTS
from analysis.schema import make_synthetic_results
from analysis.tables import decision_table, design_space_table, win_rank_table

_TASKS = ("classification", "regression")


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a tidy results table from a .parquet or .csv file."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported results format: {path.suffix} (use .parquet/.csv)")


def regenerate(results: pd.DataFrame, outdir: str | Path) -> list[Path]:
    """Write all figures, tables, and the decision guide into ``outdir``.

    Only the tasks actually present in ``results`` are rendered (in canonical
    order), so a classification-only or regression-only results file works.
    """
    if results.empty:
        raise ValueError("results is empty — nothing to regenerate.")
    present = set(results["task"].unique())
    tasks = [t for t in _TASKS if t in present]
    if not tasks:
        raise ValueError(
            f"results has no recognized task; found {sorted(present)}, "
            f"expected some of {list(_TASKS)}."
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for task in tasks:
        written.append(cd_diagram(results, task, outdir / f"cd_{task}.png"))
        written.append(grid_heatmap(results, task, outdir / f"grid_{task}.png"))
        written.append(auc_k_barchart(results, task, outdir / f"auc_k_{task}.png"))
        written.append(stability_accuracy_scatter(results, task, outdir / f"stability_{task}.png"))
        tex_path = outdir / f"win_rank_{task}.tex"
        tex_path.write_text(win_rank_table(results, task), encoding="utf-8")
        written.append(tex_path)

    design_path = outdir / "design_space.tex"
    design_path.write_text(design_space_table(), encoding="utf-8")
    written.append(design_path)

    guide = build_decision_guide(results)
    guide_tex = outdir / "decision_guide.tex"
    guide_tex.write_text(decision_table(guide), encoding="utf-8")
    written.append(guide_tex)

    guide_json = outdir / "decision_guide.json"
    serializable = {f"{task}|{regime}": v for (task, regime), v in guide.items()}
    guide_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    written.append(guide_json)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate OpenMRMR paper artifacts.")
    parser.add_argument(
        "--results",
        default=None,
        help="Path to tidy results .parquet/.csv (default: synthetic).",
    )
    parser.add_argument(
        "--outdir",
        default=str(ARTIFACTS),
        help=f"Output directory for figures/tables (default: {ARTIFACTS}).",
    )
    args = parser.parse_args(argv)
    results = make_synthetic_results() if args.results is None else load_results(args.results)
    written = regenerate(results, args.outdir)
    print(f"Wrote {len(written)} artifacts to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
