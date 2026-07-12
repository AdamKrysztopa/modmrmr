"""Tests for the ``mechanism.run_mi_comparison`` CLI.

Only TINY configs are exercised here (the full study is a controller-run later
phase, executed via the module's ``__main__`` entry point, not by these tests).
"""

from __future__ import annotations

import pandas as pd

from mechanism.mi_comparison import MI_COMPARISON_COLUMNS
from mechanism.run_mi_comparison import build_parser, main


def _read_results(out_path):
    if out_path.exists():
        return pd.read_parquet(out_path) if out_path.suffix == ".parquet" else pd.read_csv(out_path)
    csv_fallback = out_path.with_suffix(".csv")
    assert csv_fallback.exists(), f"neither {out_path} nor its csv fallback exists"
    return pd.read_csv(csv_fallback)


def test_build_parser_defaults():
    parser = build_parser()
    ns = parser.parse_args([])
    assert ns.datasets is None
    assert ns.benchmark_datasets == ["none"]
    assert ns.ks == [1, 2, 3, 5, 8, 10]
    assert ns.seeds == [0, 1, 2]
    assert ns.out == "results/mi_comparison.parquet"
    assert ns.figures_dir == "results/figures"
    assert ns.jobs == -1
    assert ns.fresh is False
    assert ns.list is False


def test_list_returns_zero_without_required_args(capsys):
    rc = main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mi_reg_k3" in out
    assert "mi_reg_k16" in out


def test_tiny_end_to_end_writes_results_summary_winner_and_figures(tmp_path):
    out = tmp_path / "mi.parquet"
    figures_dir = tmp_path / "figs"
    rc = main(
        [
            "--datasets",
            "linear_control_clf",
            "--benchmark-datasets",
            "none",
            "--estimators",
            "mi_reg_k3",
            "mi_reg_k16",
            "mixed_ksg",
            "--ks",
            "3",
            "5",
            "--seeds",
            "0",
            "--out",
            str(out),
            "--figures-dir",
            str(figures_dir),
            "--jobs",
            "1",
        ]
    )
    assert rc == 0

    df = _read_results(out)
    assert list(df.columns) == MI_COMPARISON_COLUMNS
    assert len(df) > 0

    summary_path = out.with_name(f"{out.stem}_summary.csv")
    winner_path = out.with_name(f"{out.stem}_winner.csv")
    assert summary_path.exists()
    assert winner_path.exists()

    winner_df = pd.read_csv(winner_path)
    assert list(winner_df.columns) == [
        "estimator",
        "mean_ap_golden",
        "mean_ap_nonlinear",
        "mean_ap_linear",
        "regresses_nonlinear",
    ]
    assert len(winner_df) > 0

    figure_files = list(figures_dir.glob("*.png"))
    assert len(figure_files) > 0


def test_unknown_dataset_or_estimator_errors():
    import pytest

    with pytest.raises(SystemExit):
        main(
            [
                "--datasets",
                "not_a_real_dataset",
                "--benchmark-datasets",
                "none",
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
                "linear_control_clf",
                "--benchmark-datasets",
                "none",
                "--estimators",
                "not_a_real_estimator",
                "--ks",
                "1",
                "--seeds",
                "0",
                "--out",
                "x.csv",
            ]
        )


def test_benchmark_datasets_other_than_none_is_skipped_with_notice(capsys, tmp_path):
    out = tmp_path / "mi.parquet"
    figures_dir = tmp_path / "figs"
    rc = main(
        [
            "--datasets",
            "linear_control_clf",
            "--benchmark-datasets",
            "breast_cancer",
            "--estimators",
            "mi_reg_k3",
            "--ks",
            "3",
            "--seeds",
            "0",
            "--out",
            str(out),
            "--figures-dir",
            str(figures_dir),
            "--jobs",
            "1",
        ]
    )
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "not implemented" in out_text.lower() or "skip" in out_text.lower()
