"""Tests for the ``mechanism.run_factorial`` CLI.

Only TINY configs are exercised here (per task-D1: build+test the CLI, do not run the
full study -- that is a later phase run by the controller).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mechanism.datasets import list_mechanism_datasets
from mechanism.factorial import CANONICAL_NAMED, FULL_FACTORIAL
from mechanism.factorial_protocol import FACTORIAL_COLUMNS
from mechanism.run_factorial import build_parser, main


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
    assert ns.benchmark_datasets == ["breast_cancer", "synthetic_clf", "diabetes", "friedman1"]
    assert ns.specs is None
    assert ns.ks == [1, 2, 3, 5, 8, 10]
    assert ns.thresholds == [0.0, 0.05, 0.1, 0.2]
    assert ns.seeds == [0, 1, 2]
    assert ns.out == "results/factorial.parquet"
    assert ns.figures_dir == "results/figures"
    assert ns.jobs == -1
    assert ns.list is False


def test_list_returns_zero_without_required_args(capsys):
    # --list must short-circuit before any other requirement, so the documented
    # `python -m mechanism.run_factorial --list` invocation works standalone.
    rc = main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    for spec in FULL_FACTORIAL:
        assert spec.label in out
    for name in CANONICAL_NAMED:
        assert name in out
    for name in list_mechanism_datasets():
        assert name in out
    for name in ["breast_cancer", "synthetic_clf", "diabetes", "friedman1"]:
        assert name in out


def test_tiny_end_to_end_golden_only(tmp_path):
    out = tmp_path / "f.parquet"
    figures_dir = tmp_path / "figs"
    rc = main(
        [
            "--datasets",
            "quotient_trap_reg",
            "--benchmark-datasets",
            "none",
            "--specs",
            "f|pearson_abs|multiplicative|max",
            "f|pearson_abs|quotient|mean",
            "--ks",
            "1",
            "2",
            "--thresholds",
            "0.0",
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
    assert list(df.columns) == FACTORIAL_COLUMNS
    assert len(df) > 0
    assert set(df["dataset"].unique()) == {"quotient_trap_reg"}

    summary_path = out.with_name(f"{out.stem}_summary.csv")
    assert summary_path.exists()
    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) > 0

    decision_guide_path = out.with_name(f"{out.stem}_decision_guide.csv")
    assert decision_guide_path.exists()
    decision_guide_df = pd.read_csv(decision_guide_path)
    assert len(decision_guide_df) > 0

    assert figures_dir.exists()
    figure_files = list(figures_dir.glob("*.png"))
    assert len(figure_files) > 0


def test_benchmark_path_writes_nan_recovery_and_finite_downstream(tmp_path):
    out = tmp_path / "b.parquet"
    figures_dir = tmp_path / "figs"
    rc = main(
        [
            "--datasets",
            "none",
            "--benchmark-datasets",
            "breast_cancer",
            "--specs",
            "ModMRMR",
            "--ks",
            "2",
            "--thresholds",
            "0.0",
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
    assert list(df.columns) == FACTORIAL_COLUMNS
    assert len(df) > 0
    assert set(df["dataset"].unique()) == {"breast_cancer"}

    for col in ["precision", "recall", "f1", "redundancy_rate", "noise_rate"]:
        assert df[col].isna().all(), f"benchmark rows should have NaN {col}"

    downstream = df["downstream_score"].dropna()
    assert not downstream.empty
    assert np.isfinite(downstream).all()


def test_parquet_engine_fallback_csv_out(tmp_path):
    # Exercises the non-parquet-suffix write branch directly; when a parquet engine
    # is genuinely unavailable, main() falls back to writing this same csv path.
    out = tmp_path / "c.csv"
    figures_dir = tmp_path / "figs"
    rc = main(
        [
            "--datasets",
            "quotient_trap_reg",
            "--benchmark-datasets",
            "none",
            "--specs",
            "ModMRMR",
            "--ks",
            "1",
            "--thresholds",
            "0.0",
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
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == FACTORIAL_COLUMNS
    assert len(df) > 0


def test_unknown_dataset_or_spec_errors():
    with pytest.raises(SystemExit):
        main(
            [
                "--datasets",
                "not_a_real_dataset",
                "--benchmark-datasets",
                "none",
                "--specs",
                "ModMRMR",
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
                "quotient_trap_reg",
                "--benchmark-datasets",
                "none",
                "--specs",
                "not_a_real_spec",
                "--ks",
                "1",
                "--seeds",
                "0",
                "--out",
                "x.csv",
            ]
        )
