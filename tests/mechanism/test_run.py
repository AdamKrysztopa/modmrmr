from __future__ import annotations

import pandas as pd
import pytest

from mechanism.datasets import list_mechanism_datasets
from mechanism.protocol import MECHANISM_COLUMNS
from mechanism.run import build_parser, main


def test_build_parser_defaults():
    parser = build_parser()
    ns = parser.parse_args([])
    assert ns.datasets is None
    assert ns.cells is None
    assert ns.ks == [1, 2, 5, 10]
    assert ns.thresholds is None
    assert ns.seeds == [0, 1, 2]
    assert ns.out == "results/mechanism.parquet"
    assert ns.figures_dir == "results/figures"
    assert ns.list is False


def test_build_parser_parses_all_options():
    parser = build_parser()
    ns = parser.parse_args(
        [
            "--datasets",
            "parabola",
            "radial",
            "--cells",
            "MID",
            "skb_f",
            "--ks",
            "1",
            "2",
            "--thresholds",
            "0.1",
            "0.2",
            "--seeds",
            "0",
            "1",
            "--out",
            "x.csv",
            "--figures-dir",
            "figs",
        ]
    )
    assert ns.datasets == ["parabola", "radial"]
    assert ns.cells == ["MID", "skb_f"]
    assert ns.ks == [1, 2]
    assert ns.thresholds == [0.1, 0.2]
    assert ns.seeds == [0, 1]
    assert ns.out == "x.csv"
    assert ns.figures_dir == "figs"


def test_list_returns_zero_without_required_args(capsys):
    # --list must short-circuit before any datasets/cells/out requirement, so the
    # documented `python -m mechanism.run --list` invocation works standalone.
    rc = main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "datasets:" in out
    assert "cells:" in out
    for name in list_mechanism_datasets():
        assert name in out


def test_main_writes_results_file(tmp_path):
    out = tmp_path / "m.parquet"
    figures_dir = tmp_path / "figures"
    rc = main(
        [
            "--datasets",
            "parabola",
            "--cells",
            "MID",
            "--ks",
            "1",
            "2",
            "--seeds",
            "0",
            "--out",
            str(out),
            "--figures-dir",
            str(figures_dir),
        ]
    )
    assert rc == 0

    if out.exists():
        df = pd.read_parquet(out)
    else:
        # Fall back to csv if the parquet engine isn't available under uv.
        csv_out = out.with_suffix(".csv")
        assert csv_out.exists()
        df = pd.read_csv(csv_out)

    assert list(df.columns) == MECHANISM_COLUMNS
    assert len(df) > 0

    summary_path = out.with_name(f"{out.stem}_summary.csv")
    assert summary_path.exists()
    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) > 0

    assert figures_dir.exists()
    figure_files = list(figures_dir.glob("*.png"))
    assert len(figure_files) > 0


def test_unknown_cell_or_dataset_errors():
    with pytest.raises(SystemExit):
        main(
            [
                "--datasets",
                "not_a_real_dataset",
                "--cells",
                "MID",
                "--ks",
                "1",
                "--seeds",
                "0",
                "--out",
                "x.csv",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--datasets",
                "parabola",
                "--cells",
                "not_a_real_cell",
                "--ks",
                "1",
                "--seeds",
                "0",
                "--out",
                "x.csv",
            ]
        )
