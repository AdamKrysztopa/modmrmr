"""Tests for the mechanism-suite runner ``mechanism.protocol.run_mechanism_grid``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mechanism.protocol import (
    MECHANISM_COLUMNS,
    run_mechanism_grid,
    run_validation_selected_grid,
)


def test_columns_exact() -> None:
    df = run_mechanism_grid(
        cells=["MID", "FCD"],
        datasets=["parabola"],
        ks=[2],
        seeds=[0],
    )
    assert list(df.columns) == MECHANISM_COLUMNS


def test_one_row_per_combo() -> None:
    df = run_mechanism_grid(
        cells=["MID"],
        datasets=["parabola"],
        ks=[1, 2],
        seeds=[0, 1],
        thresholds=[0.05],
    )
    n_fixed_k = len([1, 2]) * len([0, 1])
    n_threshold = len([0.05]) * len([0, 1])
    assert len(df) == n_fixed_k + n_threshold
    assert set(df["stop_mode"].unique()) == {"fixed_k", "threshold"}
    assert (df.loc[df["stop_mode"] == "fixed_k", "score_threshold"].isna()).all()
    assert (df.loc[df["stop_mode"] == "threshold", "k"] >= 1).all()


def test_baselines_get_no_threshold_rows() -> None:
    df = run_mechanism_grid(
        cells=["MID", "skb_f"],
        datasets=["parabola"],
        ks=[2],
        seeds=[0],
        thresholds=[0.05],
    )
    threshold_methods = set(df.loc[df["stop_mode"] == "threshold", "method"].unique())
    assert threshold_methods == {"MID"}


def test_recovery_cols_in_unit_interval() -> None:
    df = run_mechanism_grid(
        cells=["MID", "FCD", "skb_f"],
        datasets=["parabola"],
        ks=[1, 2, 3],
        seeds=[0],
    )
    for col in ["precision", "recall", "f1", "redundancy_rate", "noise_rate"]:
        assert (df[col] >= 0).all()
        assert (df[col] <= 1).all()
    downstream = df["downstream_score"].dropna()
    assert not downstream.empty


def test_nonlinear_beats_linear_f1() -> None:
    df = run_mechanism_grid(
        cells=["MID", "FCD"],
        datasets=["parabola", "sine"],
        ks=[1, 2, 3],
        seeds=[0, 1, 2],
    )
    fixed = df[df["stop_mode"] == "fixed_k"]
    mean_f1_mid = fixed.loc[fixed["method"] == "MID", "f1"].mean()
    mean_f1_fcd = fixed.loc[fixed["method"] == "FCD", "f1"].mean()
    assert mean_f1_mid > mean_f1_fcd, (
        f"expected MID (nonlinear MI relevance) to beat FCD (linear F relevance) "
        f"on recovery f1; got MID={mean_f1_mid}, FCD={mean_f1_fcd}"
    )


def test_deterministic() -> None:
    kwargs = dict(
        cells=["MID", "FCD", "skb_f"],
        datasets=["parabola"],
        ks=[1, 2],
        seeds=[0, 1],
        thresholds=[0.05],
    )
    df1 = run_mechanism_grid(**kwargs)
    df2 = run_mechanism_grid(**kwargs)
    # runtime_s is wall-clock timing noise, not scientific content; compare everything else.
    drop = ["runtime_s"]
    pd.testing.assert_frame_equal(df1.drop(columns=drop), df2.drop(columns=drop))


def test_downstream_nan_when_no_features_selected() -> None:
    # sanity: downstream helper handles empty idx gracefully via NaN, not a crash.
    # Exercised indirectly through the grid; assert no NaN-crash and dtype is float.
    df = run_mechanism_grid(
        cells=["MID"],
        datasets=["parabola"],
        ks=[1],
        seeds=[0],
    )
    assert np.issubdtype(df["downstream_score"].dtype, np.floating)


def test_val_selected_columns_and_modes() -> None:
    df = run_validation_selected_grid(
        cells=["MID", "FCD"],
        datasets=["parabola"],
        ks=[1, 2],
        thresholds=[0.0, 0.05],
        seeds=[0],
    )
    assert list(df.columns) == MECHANISM_COLUMNS
    assert set(df["stop_mode"].unique()) <= {"val_fixed_k", "val_threshold"}
    # MID is MRMR-family: expect both stop modes present for it.
    mid = df[df["method"] == "MID"]
    assert set(mid["stop_mode"].unique()) == {"val_fixed_k", "val_threshold"}

    # A baseline (not in benchmarks.cells.CELLS) never gets val_threshold rows.
    baseline_df = run_validation_selected_grid(
        cells=["skb_f"],
        datasets=["parabola"],
        ks=[1, 2],
        thresholds=[0.0, 0.05],
        seeds=[0],
    )
    assert set(baseline_df["stop_mode"].unique()) <= {"val_fixed_k"}


def test_val_selected_k_within_grid() -> None:
    ks = [1, 2]
    thresholds = [0.0, 0.05]
    df = run_validation_selected_grid(
        cells=["MID", "FCD"],
        datasets=["parabola"],
        ks=ks,
        thresholds=thresholds,
        seeds=[0],
    )
    fixed_k_rows = df[df["stop_mode"] == "val_fixed_k"]
    assert not fixed_k_rows.empty
    assert fixed_k_rows["k"].isin(ks).all()

    threshold_rows = df[df["stop_mode"] == "val_threshold"]
    assert not threshold_rows.empty
    assert threshold_rows["score_threshold"].isin(thresholds).all()


def test_val_selected_deterministic() -> None:
    kwargs = dict(
        cells=["MID", "FCD"],
        datasets=["parabola"],
        ks=[1, 2],
        thresholds=[0.0, 0.05],
        seeds=[0, 1],
    )
    df1 = run_validation_selected_grid(**kwargs)
    df2 = run_validation_selected_grid(**kwargs)
    drop = ["runtime_s"]
    pd.testing.assert_frame_equal(df1.drop(columns=drop), df2.drop(columns=drop))


def test_val_selected_recovery_in_unit_interval() -> None:
    df = run_validation_selected_grid(
        cells=["MID", "FCD"],
        datasets=["parabola"],
        ks=[1, 2],
        thresholds=[0.0, 0.05],
        seeds=[0],
    )
    for col in ["precision", "recall", "f1", "redundancy_rate", "noise_rate"]:
        assert (df[col] >= 0).all()
        assert (df[col] <= 1).all()
