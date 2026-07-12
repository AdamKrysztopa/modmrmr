import pandas as pd
import pytest

pytest.importorskip("mrmr")  # mrmr-selection package (benchmarks dependency group)

from mechanism.baseline_comparison import BASELINE_COLUMNS, run_baseline_grid
from mechanism.run_baseline_comparison import build_parser, main


def test_parser_defaults():
    ns = build_parser().parse_args([])
    assert ns.ks == [3, 5, 10]
    assert ns.seeds == [0, 1, 2]
    assert ns.out == "results/baseline_comparison.csv"


def test_tiny_grid_external_and_internal_rows():
    df = run_baseline_grid(
        methods=["mrmr_selection_classif", "ModMRMR"],
        datasets=["linear_control_clf"],
        ks=[3],
        seeds=[0],
    )
    assert list(df.columns) == BASELINE_COLUMNS
    assert set(df["source"]) == {"external", "modmrmr"}
    assert df["recovery_f1"].between(0, 1).all()
    assert (df["runtime_s"] > 0).all()


def test_cli_end_to_end_writes_csv(tmp_path):
    out = tmp_path / "baseline.csv"
    rc = main(
        [
            "--methods",
            "mrmr_selection_classif",
            "--datasets",
            "linear_control_clf",
            "--ks",
            "3",
            "--seeds",
            "0",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == BASELINE_COLUMNS
    assert len(df) >= 1
