"""Tests for ``mechanism.factorial_protocol.run_factorial_grid``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mechanism.factorial import CANONICAL_NAMED
from mechanism.factorial_protocol import (
    FACTORIAL_COLUMNS,
    run_downstream_only_grid,
    run_factorial_grid,
)

_SPECS = [CANONICAL_NAMED["FCQ"], CANONICAL_NAMED["ModMRMR"]]
_KWARGS = dict(
    specs=_SPECS,
    datasets=["quotient_trap_reg"],
    ks=[1, 2, 3],
    thresholds=[0.0, 0.05],
    seeds=[0],
)


def test_columns_exact() -> None:
    df = run_factorial_grid(**_KWARGS)
    assert list(df.columns) == FACTORIAL_COLUMNS


def test_recovery_and_ranking_cols_in_unit_interval() -> None:
    df = run_factorial_grid(**_KWARGS)
    for col in [
        "precision",
        "recall",
        "f1",
        "redundancy_rate",
        "noise_rate",
        "average_precision",
        "roc_auc",
    ]:
        vals = df[col].dropna()
        assert not vals.empty, f"{col} is all-NaN; something errored out entirely"
        assert (vals >= 0).all(), f"{col} has values below 0"
        assert (vals <= 1).all(), f"{col} has values above 1"


def test_no_unexpected_errors() -> None:
    df = run_factorial_grid(**_KWARGS)
    errored = df[df["error"].notna()]
    assert errored.empty, f"unexpected errors:\n{errored[['spec', 'stop_mode', 'error']]}"


def test_all_four_stop_modes_present() -> None:
    df = run_factorial_grid(**_KWARGS)
    assert set(df["stop_mode"].unique()) == {
        "fixed_k",
        "threshold",
        "val_fixed_k",
        "val_threshold",
    }


def test_modmrmr_noise_rate_lt_quotient_on_quotient_trap_threshold() -> None:
    """The load-bearing niche, isolated where the separation is actually real:
    threshold mode. Holding relevance+redundancy fixed (both f|pearson_abs) and
    varying only operator+aggregation, ModMRMR (multiplicative+max) picks
    STRICTLY less noise than FCQ (quotient) on quotient_trap_reg at
    score_threshold=0.05. FCQ's near-zero aggregated-redundancy denominator
    inflates the score of independent distractor columns so it never stops
    early (it selects all 13 features at every threshold); ModMRMR's
    multiplicative veto correctly saturates and stops selecting once
    redundancy accumulates, giving it strictly lower noise_rate at the same
    threshold. Both specs share f|pearson_abs, so this is the honest
    apples-to-apples test of the operator+aggregation difference.

    This must be threshold mode, not fixed_k: with k in {1,2,3} and only 3
    informative features, both operators fill the pick with pure signal and
    noise_rate ties at 0.0 for both specs -- an assertion restricted to
    fixed_k rows would still pass even if ModMRMR's multiplicative veto were
    broken/reverted to a plain quotient. Threshold mode gives the selector
    room (13 candidate features, no k cap) to actually diverge.
    """
    df = run_factorial_grid(**_KWARGS)
    threshold_rows = df[(df["stop_mode"] == "threshold") & (df["score_threshold"] == 0.05)]
    modmrmr_noise = threshold_rows.loc[
        threshold_rows["spec"] == CANONICAL_NAMED["ModMRMR"].label, "noise_rate"
    ].mean()
    fcq_noise = threshold_rows.loc[
        threshold_rows["spec"] == CANONICAL_NAMED["FCQ"].label, "noise_rate"
    ].mean()
    assert modmrmr_noise < fcq_noise, (
        f"expected ModMRMR noise_rate < FCQ noise_rate on quotient_trap_reg at "
        f"threshold=0.05 (the niche's real, load-bearing separation); "
        f"got ModMRMR={modmrmr_noise}, FCQ={fcq_noise}"
    )


def test_modmrmr_noise_rate_le_quotient_on_quotient_trap_fixed_k() -> None:
    """Secondary non-regression guard, not the primary claim: on fixed_k rows
    (k in {1,2,3}) both specs tie at noise_rate=0.0 on this dataset because
    only 3 slots and 3 informative features leave no room to diverge -- see
    the threshold-mode test above for the actual strict separation. This just
    guards that fixed_k never regresses to ModMRMR being *worse* than FCQ."""
    df = run_factorial_grid(**_KWARGS)
    fixed = df[df["stop_mode"] == "fixed_k"]
    modmrmr_noise = fixed.loc[
        fixed["spec"] == CANONICAL_NAMED["ModMRMR"].label, "noise_rate"
    ].mean()
    fcq_noise = fixed.loc[fixed["spec"] == CANONICAL_NAMED["FCQ"].label, "noise_rate"].mean()
    assert modmrmr_noise <= fcq_noise, (
        f"expected ModMRMR noise_rate <= FCQ noise_rate on quotient_trap_reg; "
        f"got ModMRMR={modmrmr_noise}, FCQ={fcq_noise}"
    )


def test_deterministic() -> None:
    df1 = run_factorial_grid(**_KWARGS)
    df2 = run_factorial_grid(**_KWARGS)
    drop = ["runtime_s"]
    pd.testing.assert_frame_equal(df1.drop(columns=drop), df2.drop(columns=drop))


def test_downstream_score_is_float_column() -> None:
    df = run_factorial_grid(**_KWARGS)
    assert np.issubdtype(df["downstream_score"].dtype, np.floating)


def test_include_downstream_false_leaves_downstream_nan_but_still_runs() -> None:
    df = run_factorial_grid(**{**_KWARGS, "include_downstream": False})
    assert df["downstream_score"].isna().all()
    # recovery/ranking metrics are still computed even without downstream scoring.
    assert not df["f1"].dropna().empty


def test_empty_thresholds_skips_threshold_stop_modes() -> None:
    df = run_factorial_grid(**{**_KWARGS, "thresholds": None})
    assert set(df["stop_mode"].unique()) == {"fixed_k", "val_fixed_k"}


# --------------------------------------------------------------------------- #
# run_downstream_only_grid — benchmark datasets have no GroundTruth, so recovery/
# ranking metrics are NaN; only downstream_score is populated.
# --------------------------------------------------------------------------- #
_BENCHMARK_SPECS = [CANONICAL_NAMED["ModMRMR"]]
_BENCHMARK_KWARGS = dict(
    specs=_BENCHMARK_SPECS,
    datasets=["breast_cancer"],
    ks=[2],
    thresholds=None,
    seeds=[0],
)


def test_downstream_only_columns_exact() -> None:
    df = run_downstream_only_grid(**_BENCHMARK_KWARGS)
    assert list(df.columns) == FACTORIAL_COLUMNS


def test_downstream_only_recovery_cols_are_nan() -> None:
    df = run_downstream_only_grid(**_BENCHMARK_KWARGS)
    for col in [
        "precision",
        "recall",
        "f1",
        "redundancy_rate",
        "noise_rate",
        "average_precision",
        "roc_auc",
    ]:
        assert df[col].isna().all(), f"{col} should be all-NaN for a benchmark (no GroundTruth)"


def test_downstream_only_downstream_score_finite() -> None:
    df = run_downstream_only_grid(**_BENCHMARK_KWARGS)
    scores = df["downstream_score"].dropna()
    assert not scores.empty, "downstream_score should be populated for a successful fit"
    assert np.isfinite(scores).all()
    assert (scores <= 1).all()


def test_downstream_only_deterministic() -> None:
    df1 = run_downstream_only_grid(**_BENCHMARK_KWARGS)
    df2 = run_downstream_only_grid(**_BENCHMARK_KWARGS)
    pd.testing.assert_frame_equal(df1.drop(columns=["runtime_s"]), df2.drop(columns=["runtime_s"]))
