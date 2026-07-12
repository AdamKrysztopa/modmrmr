from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from mechanism.factorial import CANONICAL_NAMED
from mechanism.factorial_protocol import FACTORIAL_COLUMNS
from mechanism.figures import (
    _features_vs_threshold_summary,
    decision_guide_table,
    factorial_summary,
    features_vs_threshold,
    fixed_k_vs_threshold,
    linear_vs_nonlinear_gap,
    measure_family,
    mechanism_summary,
    mi_ap_by_dependence,
    mi_comparison_summary,
    mi_ranking_leaderboard,
    mi_winner,
    operator_aggregation_heatmap,
    ranking_leaderboard,
    recovery_vs_k,
)
from mechanism.mi_comparison import MI_COMPARISON_COLUMNS
from mechanism.protocol import MECHANISM_COLUMNS


def _row(**kwargs) -> dict:
    base = {
        "dataset": "parabola",
        "dependence": "nonlinear",
        "task": "regression",
        "method": "MID",
        "operator": "difference",
        "aggregation": "mean",
        "relevance": "mutual_info_sklearn",
        "redundancy": "mutual_info_sklearn",
        "stop_mode": "fixed_k",
        "k": 2,
        "score_threshold": float("nan"),
        "seed": 0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "redundancy_rate": 0.0,
        "noise_rate": 0.0,
        "downstream_score": 0.5,
        "runtime_s": 0.01,
    }
    base.update(kwargs)
    return base


@pytest.fixture
def synthetic_mechanism_df() -> pd.DataFrame:
    rows = [
        # parabola (nonlinear dataset), nonlinear measures, fixed_k, k=1 and k=2.
        _row(
            dataset="parabola",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="fixed_k",
            k=1,
            f1=0.4,
            precision=0.4,
            recall=0.4,
            seed=0,
        ),
        _row(
            dataset="parabola",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="fixed_k",
            k=1,
            f1=0.6,
            precision=0.6,
            recall=0.6,
            seed=1,
        ),
        _row(
            dataset="parabola",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="fixed_k",
            k=2,
            f1=0.8,
            precision=0.8,
            recall=0.8,
            seed=0,
        ),
        _row(
            dataset="parabola",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="fixed_k",
            k=2,
            f1=1.0,
            precision=1.0,
            recall=1.0,
            seed=1,
        ),
        # parabola, linear measures, fixed_k -- should score much worse.
        _row(
            dataset="parabola",
            relevance="f_regression",
            redundancy="pearson_abs",
            stop_mode="fixed_k",
            k=1,
            f1=0.1,
            precision=0.1,
            recall=0.1,
            seed=0,
        ),
        _row(
            dataset="parabola",
            relevance="f_regression",
            redundancy="pearson_abs",
            stop_mode="fixed_k",
            k=2,
            f1=0.2,
            precision=0.2,
            recall=0.2,
            seed=0,
        ),
        # parabola, nonlinear measures, threshold stop mode.
        _row(
            dataset="parabola",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="threshold",
            k=2,
            score_threshold=0.1,
            f1=0.9,
            precision=0.9,
            recall=0.9,
            seed=0,
        ),
        _row(
            dataset="parabola",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="threshold",
            k=2,
            score_threshold=0.1,
            f1=0.7,
            precision=0.7,
            recall=0.7,
            seed=1,
        ),
        # linear_control_reg (linear dataset), nonlinear measures, fixed_k.
        _row(
            dataset="linear_control_reg",
            dependence="linear",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="fixed_k",
            k=1,
            f1=0.5,
            precision=0.5,
            recall=0.5,
            seed=0,
        ),
        _row(
            dataset="linear_control_reg",
            dependence="linear",
            relevance="mutual_info_sklearn",
            redundancy="mutual_info_sklearn",
            stop_mode="fixed_k",
            k=2,
            f1=0.6,
            precision=0.6,
            recall=0.6,
            seed=0,
        ),
        # linear_control_reg, linear measures, fixed_k -- near-identical to nonlinear here.
        _row(
            dataset="linear_control_reg",
            dependence="linear",
            relevance="f_regression",
            redundancy="pearson_abs",
            stop_mode="fixed_k",
            k=1,
            f1=0.5,
            precision=0.5,
            recall=0.5,
            seed=0,
        ),
        _row(
            dataset="linear_control_reg",
            dependence="linear",
            relevance="f_regression",
            redundancy="pearson_abs",
            stop_mode="fixed_k",
            k=2,
            f1=0.6,
            precision=0.6,
            recall=0.6,
            seed=0,
        ),
        # linear_control_reg, linear measures, threshold.
        _row(
            dataset="linear_control_reg",
            dependence="linear",
            relevance="f_regression",
            redundancy="pearson_abs",
            stop_mode="threshold",
            k=2,
            score_threshold=0.1,
            f1=0.55,
            precision=0.55,
            recall=0.55,
            seed=0,
        ),
    ]
    df = pd.DataFrame(rows)
    assert list(df.columns) == MECHANISM_COLUMNS
    return df


def _frow(**kwargs) -> dict:
    """Build one FACTORIAL_COLUMNS-shaped row, defaulting to a ModMRMR /
    quotient_trap_reg / fixed_k cell. Field order matches FACTORIAL_COLUMNS."""
    base = {
        "dataset": "quotient_trap_reg",
        "dependence": "linear",
        "task": "regression",
        "spec": CANONICAL_NAMED["ModMRMR"].label,
        "relevance": "f_regression",
        "redundancy": "pearson_abs",
        "operator": "multiplicative",
        "aggregation": "max",
        "stop_mode": "fixed_k",
        "k": 3,
        "score_threshold": float("nan"),
        "seed": 0,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
        "redundancy_rate": 0.0,
        "noise_rate": 0.0,
        "average_precision": 0.9,
        "roc_auc": 0.9,
        "downstream_score": 0.8,
        "runtime_s": 0.01,
        "error": None,
    }
    base.update(kwargs)
    return base


@pytest.fixture
def factorial_df() -> pd.DataFrame:
    modmrmr = CANONICAL_NAMED["ModMRMR"].label
    mid = CANONICAL_NAMED["MID"].label
    miq = CANONICAL_NAMED["MIQ"].label
    rows = [
        # --- quotient_trap_reg (linear dependence, true_k=3): all four stop modes.
        _frow(
            spec=modmrmr,
            relevance="f_regression",
            redundancy="pearson_abs",
            operator="multiplicative",
            aggregation="max",
            stop_mode="fixed_k",
            k=3,
            f1=0.90,
            precision=0.90,
            recall=0.90,
            redundancy_rate=0.0,
            noise_rate=0.0,
            average_precision=0.95,
            roc_auc=0.95,
            downstream_score=0.80,
        ),
        _frow(
            spec=modmrmr,
            relevance="f_regression",
            redundancy="pearson_abs",
            operator="multiplicative",
            aggregation="max",
            stop_mode="threshold",
            k=3.0,
            score_threshold=0.05,
            f1=0.85,
            precision=0.85,
            recall=0.85,
            redundancy_rate=0.0,
            noise_rate=0.05,
            average_precision=0.90,
            roc_auc=0.90,
            downstream_score=0.78,
        ),
        _frow(
            spec=modmrmr,
            relevance="f_regression",
            redundancy="pearson_abs",
            operator="multiplicative",
            aggregation="max",
            stop_mode="val_fixed_k",
            k=3,
            f1=0.88,
            precision=0.88,
            recall=0.88,
            redundancy_rate=0.0,
            noise_rate=0.0,
            average_precision=0.92,
            roc_auc=0.92,
            downstream_score=0.79,
        ),
        _frow(
            spec=modmrmr,
            relevance="f_regression",
            redundancy="pearson_abs",
            operator="multiplicative",
            aggregation="max",
            stop_mode="val_threshold",
            k=3.0,
            score_threshold=0.05,
            f1=0.83,
            precision=0.83,
            recall=0.83,
            redundancy_rate=0.0,
            noise_rate=0.05,
            average_precision=0.88,
            roc_auc=0.88,
            downstream_score=0.77,
        ),
        _frow(
            spec=mid,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="difference",
            aggregation="mean",
            stop_mode="fixed_k",
            k=3,
            f1=0.60,
            precision=0.60,
            recall=0.60,
            redundancy_rate=0.1,
            noise_rate=0.2,
            average_precision=0.55,
            roc_auc=0.55,
            downstream_score=0.50,
        ),
        _frow(
            spec=mid,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="difference",
            aggregation="mean",
            stop_mode="threshold",
            k=4.0,
            score_threshold=0.05,
            f1=0.55,
            precision=0.55,
            recall=0.55,
            redundancy_rate=0.1,
            noise_rate=0.25,
            average_precision=0.50,
            roc_auc=0.50,
            downstream_score=0.48,
        ),
        _frow(
            spec=miq,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="quotient",
            aggregation="mean",
            stop_mode="fixed_k",
            k=3,
            f1=0.50,
            precision=0.50,
            recall=0.50,
            redundancy_rate=0.1,
            noise_rate=0.3,
            average_precision=0.45,
            roc_auc=0.45,
            downstream_score=0.40,
        ),
        _frow(
            spec=miq,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="quotient",
            aggregation="mean",
            stop_mode="threshold",
            k=5.0,
            score_threshold=0.05,
            f1=0.40,
            precision=0.40,
            recall=0.40,
            redundancy_rate=0.15,
            noise_rate=0.4,
            average_precision=0.35,
            roc_auc=0.35,
            downstream_score=0.30,
        ),
        # --- parabola (nonlinear dependence, true_k=1): fixed_k sweep k=1,2 only.
        _frow(
            dataset="parabola",
            dependence="nonlinear",
            spec=modmrmr,
            relevance="f_regression",
            redundancy="pearson_abs",
            operator="multiplicative",
            aggregation="max",
            stop_mode="fixed_k",
            k=1,
            f1=0.9,
            precision=0.9,
            recall=0.9,
            redundancy_rate=0.0,
            noise_rate=0.0,
            average_precision=0.85,
            roc_auc=0.85,
            downstream_score=0.70,
        ),
        _frow(
            dataset="parabola",
            dependence="nonlinear",
            spec=modmrmr,
            relevance="f_regression",
            redundancy="pearson_abs",
            operator="multiplicative",
            aggregation="max",
            stop_mode="fixed_k",
            k=2,
            f1=0.6,
            precision=0.6,
            recall=0.6,
            redundancy_rate=0.2,
            noise_rate=0.1,
            average_precision=0.65,
            roc_auc=0.65,
            downstream_score=0.55,
        ),
        _frow(
            dataset="parabola",
            dependence="nonlinear",
            spec=mid,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="difference",
            aggregation="mean",
            stop_mode="fixed_k",
            k=1,
            f1=0.5,
            precision=0.5,
            recall=0.5,
            redundancy_rate=0.0,
            noise_rate=0.2,
            average_precision=0.45,
            roc_auc=0.45,
            downstream_score=0.40,
        ),
        _frow(
            dataset="parabola",
            dependence="nonlinear",
            spec=mid,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="difference",
            aggregation="mean",
            stop_mode="fixed_k",
            k=2,
            f1=0.3,
            precision=0.3,
            recall=0.3,
            redundancy_rate=0.3,
            noise_rate=0.3,
            average_precision=0.25,
            roc_auc=0.25,
            downstream_score=0.20,
        ),
        _frow(
            dataset="parabola",
            dependence="nonlinear",
            spec=miq,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="quotient",
            aggregation="mean",
            stop_mode="fixed_k",
            k=1,
            f1=0.4,
            precision=0.4,
            recall=0.4,
            redundancy_rate=0.1,
            noise_rate=0.3,
            average_precision=0.35,
            roc_auc=0.35,
            downstream_score=0.30,
        ),
        _frow(
            dataset="parabola",
            dependence="nonlinear",
            spec=miq,
            relevance="mutual_info_regression",
            redundancy="mutual_info_sklearn",
            operator="quotient",
            aggregation="mean",
            stop_mode="fixed_k",
            k=2,
            f1=0.2,
            precision=0.2,
            recall=0.2,
            redundancy_rate=0.4,
            noise_rate=0.4,
            average_precision=0.15,
            roc_auc=0.15,
            downstream_score=0.10,
        ),
    ]
    df = pd.DataFrame(rows)
    assert list(df.columns) == FACTORIAL_COLUMNS
    return df


class TestMeasureFamily:
    def test_both_mutual_info_is_nonlinear(self) -> None:
        assert measure_family("mutual_info_sklearn", "mutual_info_sklearn") == "nonlinear"

    def test_f_regression_and_pearson_is_linear(self) -> None:
        assert measure_family("f_regression", "pearson_abs") == "linear"

    def test_mixed_mutual_info_and_pearson_is_mixed(self) -> None:
        assert measure_family("mutual_info_sklearn", "pearson_abs") == "mixed"

    def test_both_distance_corr_is_nonlinear(self) -> None:
        assert measure_family("distance_corr", "distance_corr") == "nonlinear"

    def test_spearman_and_empty_is_linear(self) -> None:
        assert measure_family("spearman_abs", "") == "linear"

    def test_f_classif_alias_is_linear(self) -> None:
        assert measure_family("f_classif", "f_classif") == "linear"


class TestRecoveryVsK:
    def test_writes_nonempty_file(
        self, synthetic_mechanism_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "recovery_vs_k_parabola.png"
        result = recovery_vs_k(synthetic_mechanism_df, "parabola", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class TestRecoveryVsKFactorial:
    def test_writes_nonempty_file_for_factorial_df(
        self, factorial_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "recovery_vs_k_parabola_factorial.png"
        result = recovery_vs_k(factorial_df, "parabola", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_writes_nonempty_file_quotient_trap_reg(
        self, factorial_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "recovery_vs_k_quotient_trap_reg_factorial.png"
        result = recovery_vs_k(factorial_df, "quotient_trap_reg", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class TestLinearVsNonlinearGap:
    def test_writes_nonempty_file(
        self, synthetic_mechanism_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "linear_vs_nonlinear_gap.png"
        result = linear_vs_nonlinear_gap(synthetic_mechanism_df, out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class TestFixedKVsThreshold:
    def test_writes_nonempty_file(
        self, synthetic_mechanism_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "fixed_k_vs_threshold.png"
        result = fixed_k_vs_threshold(synthetic_mechanism_df, out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class TestMechanismSummary:
    def test_has_expected_columns(self, synthetic_mechanism_df: pd.DataFrame) -> None:
        summary = mechanism_summary(synthetic_mechanism_df)
        expected = {
            "dataset",
            "family",
            "precision",
            "recall",
            "f1",
            "redundancy_rate",
            "noise_rate",
            "downstream_score",
        }
        assert expected.issubset(set(summary.columns))

    def test_hand_checked_mean_value(self, synthetic_mechanism_df: pd.DataFrame) -> None:
        summary = mechanism_summary(synthetic_mechanism_df)
        # parabola / nonlinear rows: f1 = 0.4, 0.6, 0.8, 1.0, 0.9, 0.7 -> mean = 0.733...
        row = summary[(summary["dataset"] == "parabola") & (summary["family"] == "nonlinear")]
        assert len(row) == 1
        expected_mean_f1 = (0.4 + 0.6 + 0.8 + 1.0 + 0.9 + 0.7) / 6
        assert row["f1"].iloc[0] == pytest.approx(expected_mean_f1)

    def test_linear_control_gap_is_small(self, synthetic_mechanism_df: pd.DataFrame) -> None:
        gap_df = synthetic_mechanism_df[synthetic_mechanism_df["stop_mode"] == "fixed_k"]
        summary = mechanism_summary(gap_df)
        linear_control = summary[summary["dataset"] == "linear_control_reg"]
        nonlinear_f1 = linear_control.loc[linear_control["family"] == "nonlinear", "f1"].iloc[0]
        linear_f1 = linear_control.loc[linear_control["family"] == "linear", "f1"].iloc[0]
        assert abs(nonlinear_f1 - linear_f1) < 0.05


class TestOperatorAggregationHeatmap:
    def test_writes_nonempty_file_f1(self, factorial_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "operator_aggregation_heatmap_f1.png"
        result = operator_aggregation_heatmap(factorial_df, "f1", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_writes_nonempty_file_noise_rate(
        self, factorial_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "operator_aggregation_heatmap_noise_rate.png"
        result = operator_aggregation_heatmap(factorial_df, "noise_rate", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class TestRankingLeaderboard:
    def test_writes_nonempty_file(self, factorial_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "ranking_leaderboard.png"
        result = ranking_leaderboard(factorial_df, out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_ranked_descending_by_mean_average_precision(self, factorial_df: pd.DataFrame) -> None:
        means = (
            factorial_df.groupby("spec")["average_precision"].mean().sort_values(ascending=False)
        )
        # ModMRMR (multiplicative+max) should rank first given the fixture's values.
        assert means.index[0] == CANONICAL_NAMED["ModMRMR"].label


class TestFeaturesVsThreshold:
    def test_covers_all_present_stop_modes(self, factorial_df: pd.DataFrame) -> None:
        summary = _features_vs_threshold_summary(factorial_df, "quotient_trap_reg")
        present = set(factorial_df.loc[factorial_df["dataset"] == "quotient_trap_reg", "stop_mode"])
        assert present == {"fixed_k", "threshold", "val_fixed_k", "val_threshold"}
        assert set(summary["stop_mode"]) == present

    def test_writes_nonempty_file(self, factorial_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "features_vs_threshold_quotient_trap_reg.png"
        result = features_vs_threshold(factorial_df, "quotient_trap_reg", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class TestDecisionGuideTable:
    def test_has_expected_columns(self, factorial_df: pd.DataFrame) -> None:
        table = decision_guide_table(factorial_df)
        expected = {
            "dataset",
            "dependence",
            "best_spec_by_f1",
            "best_f1",
            "best_spec_by_noise_rate",
            "best_noise_rate",
        }
        assert expected.issubset(set(table.columns))

    def test_hand_checked_quotient_trap_reg_recommendation(
        self, factorial_df: pd.DataFrame
    ) -> None:
        table = decision_guide_table(factorial_df)
        row = table[table["dataset"] == "quotient_trap_reg"]
        assert len(row) == 1
        modmrmr = CANONICAL_NAMED["ModMRMR"].label
        assert row["best_spec_by_f1"].iloc[0] == modmrmr
        expected_f1 = (0.90 + 0.85 + 0.88 + 0.83) / 4
        assert row["best_f1"].iloc[0] == pytest.approx(expected_f1)
        assert row["best_spec_by_noise_rate"].iloc[0] == modmrmr
        expected_noise = (0.0 + 0.05 + 0.0 + 0.05) / 4
        assert row["best_noise_rate"].iloc[0] == pytest.approx(expected_noise)


class TestFactorialSummary:
    def test_has_expected_columns(self, factorial_df: pd.DataFrame) -> None:
        summary = factorial_summary(factorial_df)
        expected = {
            "spec",
            "dependence",
            "precision",
            "recall",
            "f1",
            "redundancy_rate",
            "noise_rate",
            "average_precision",
            "roc_auc",
            "downstream_score",
        }
        assert expected.issubset(set(summary.columns))

    def test_hand_checked_mean_value(self, factorial_df: pd.DataFrame) -> None:
        summary = factorial_summary(factorial_df)
        row = summary[
            (summary["spec"] == CANONICAL_NAMED["ModMRMR"].label)
            & (summary["dependence"] == "linear")
        ]
        assert len(row) == 1
        expected_f1 = (0.90 + 0.85 + 0.88 + 0.83) / 4
        assert row["f1"].iloc[0] == pytest.approx(expected_f1)
        expected_ap = (0.95 + 0.90 + 0.92 + 0.88) / 4
        assert row["average_precision"].iloc[0] == pytest.approx(expected_ap)


def _mirow(**kwargs) -> dict:
    """Build one MI_COMPARISON_COLUMNS-shaped row, defaulting to a mi_reg_k3 /
    parabola / nonlinear / seed-0 cell."""
    base = {
        "dataset": "parabola",
        "dependence": "nonlinear",
        "task": "regression",
        "estimator": "mi_reg_k3",
        "k": 3,
        "seed": 0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "redundancy_rate": 0.0,
        "noise_rate": 0.0,
        "average_precision": 0.5,
        "roc_auc": 0.5,
        "downstream_score": 0.5,
        "runtime_s": 0.01,
        "error": None,
    }
    base.update(kwargs)
    return base


@pytest.fixture
def mi_comparison_df() -> pd.DataFrame:
    """3 estimators (incumbent mi_reg_k3 + mixed_ksg + copula_mi), 2 dependence
    classes (nonlinear/linear), plus one error row (NaN metrics) to verify the
    NaN-drop-before-aggregating behaviour.

    Hand-computed means (used by the tests below):
    - mi_reg_k3:  nonlinear AP = mean(0.8, 0.6) = 0.7; linear AP = 0.9
                  golden AP = mean(0.8, 0.6, 0.9) = 0.7666...
    - mixed_ksg:  nonlinear AP = mean(0.5, 0.5) = 0.5 (clearly < 0.7 - 0.02 = 0.68
                  -> regresses_nonlinear=True); linear AP = 0.85; the seed=2
                  nonlinear row is an error row (NaN AP) and must be dropped.
                  golden AP (excluding the error row) = mean(0.5, 0.5, 0.85) = 0.6166...
    - copula_mi:  nonlinear AP = mean(0.75, 0.75) = 0.75 (clearly > 0.68
                  -> regresses_nonlinear=False); linear AP = 0.95
                  golden AP = mean(0.75, 0.75, 0.95) = 0.8166...
    -> ranking by mean_ap_golden desc: copula_mi, mi_reg_k3, mixed_ksg.
    """
    rows = [
        _mirow(estimator="mi_reg_k3", dependence="nonlinear", seed=0, average_precision=0.8),
        _mirow(estimator="mi_reg_k3", dependence="nonlinear", seed=1, average_precision=0.6),
        _mirow(estimator="mi_reg_k3", dependence="linear", seed=0, average_precision=0.9),
        _mirow(estimator="mixed_ksg", dependence="nonlinear", seed=0, average_precision=0.5),
        _mirow(estimator="mixed_ksg", dependence="nonlinear", seed=1, average_precision=0.5),
        _mirow(estimator="mixed_ksg", dependence="linear", seed=0, average_precision=0.85),
        _mirow(
            estimator="mixed_ksg",
            dependence="nonlinear",
            seed=2,
            precision=float("nan"),
            recall=float("nan"),
            f1=float("nan"),
            redundancy_rate=float("nan"),
            noise_rate=float("nan"),
            average_precision=float("nan"),
            roc_auc=float("nan"),
            downstream_score=float("nan"),
            error="RuntimeError: boom",
        ),
        _mirow(estimator="copula_mi", dependence="nonlinear", seed=0, average_precision=0.75),
        _mirow(estimator="copula_mi", dependence="nonlinear", seed=1, average_precision=0.75),
        _mirow(estimator="copula_mi", dependence="linear", seed=0, average_precision=0.95),
    ]
    df = pd.DataFrame(rows)
    assert list(df.columns) == MI_COMPARISON_COLUMNS
    return df


class TestMatplotlibBackend:
    def test_agg_backend_is_active(self) -> None:
        assert matplotlib.get_backend().lower() == "agg"


class TestMiRankingLeaderboard:
    def test_writes_nonempty_file(self, mi_comparison_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "mi_ranking_leaderboard.png"
        result = mi_ranking_leaderboard(mi_comparison_df, out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_ranked_descending_by_mean_average_precision(
        self, mi_comparison_df: pd.DataFrame
    ) -> None:
        subset = mi_comparison_df.dropna(subset=["average_precision"])
        means = subset.groupby("estimator")["average_precision"].mean().sort_values(ascending=False)
        assert means.index[0] == "copula_mi"
        assert means.index[-1] == "mixed_ksg"


class TestMiApByDependence:
    def test_writes_nonempty_file(self, mi_comparison_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "mi_ap_by_dependence.png"
        result = mi_ap_by_dependence(mi_comparison_df, out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0


class TestMiComparisonSummary:
    def test_has_expected_columns(self, mi_comparison_df: pd.DataFrame) -> None:
        summary = mi_comparison_summary(mi_comparison_df)
        expected = {
            "estimator",
            "dependence",
            "precision",
            "recall",
            "f1",
            "redundancy_rate",
            "noise_rate",
            "average_precision",
            "roc_auc",
            "downstream_score",
        }
        assert expected.issubset(set(summary.columns))

    def test_hand_checked_mean_ap_copula_mi_nonlinear(self, mi_comparison_df: pd.DataFrame) -> None:
        summary = mi_comparison_summary(mi_comparison_df)
        row = summary[
            (summary["estimator"] == "copula_mi") & (summary["dependence"] == "nonlinear")
        ]
        assert len(row) == 1
        assert row["average_precision"].iloc[0] == pytest.approx(0.75)

    def test_hand_checked_mean_ap_mixed_ksg_nonlinear_drops_error_row(
        self, mi_comparison_df: pd.DataFrame
    ) -> None:
        summary = mi_comparison_summary(mi_comparison_df)
        row = summary[
            (summary["estimator"] == "mixed_ksg") & (summary["dependence"] == "nonlinear")
        ]
        assert len(row) == 1
        # NaN-skipping mean over the two non-error rows (0.5, 0.5); the error
        # row's NaN average_precision must not turn this into NaN.
        assert row["average_precision"].iloc[0] == pytest.approx(0.5)


class TestMiWinner:
    def test_has_expected_columns_and_sorted_descending(
        self, mi_comparison_df: pd.DataFrame
    ) -> None:
        table = mi_winner(mi_comparison_df)
        expected_columns = [
            "estimator",
            "mean_ap_golden",
            "mean_ap_nonlinear",
            "mean_ap_linear",
            "regresses_nonlinear",
        ]
        assert list(table.columns) == expected_columns
        golden = list(table["mean_ap_golden"])
        assert golden == sorted(golden, reverse=True)
        assert list(table["estimator"]) == ["copula_mi", "mi_reg_k3", "mixed_ksg"]

    def test_hand_checked_golden_means(self, mi_comparison_df: pd.DataFrame) -> None:
        table = mi_winner(mi_comparison_df).set_index("estimator")
        assert table.loc["mi_reg_k3", "mean_ap_golden"] == pytest.approx((0.8 + 0.6 + 0.9) / 3)
        assert table.loc["mixed_ksg", "mean_ap_golden"] == pytest.approx((0.5 + 0.5 + 0.85) / 3)
        assert table.loc["copula_mi", "mean_ap_golden"] == pytest.approx((0.75 + 0.75 + 0.95) / 3)

    def test_regresses_nonlinear_flags(self, mi_comparison_df: pd.DataFrame) -> None:
        table = mi_winner(mi_comparison_df).set_index("estimator")
        # incumbent mi_reg_k3 nonlinear AP = 0.7; threshold = 0.7 - 0.02 = 0.68.
        assert table.loc["mi_reg_k3", "mean_ap_nonlinear"] == pytest.approx(0.7)
        assert table.loc["mi_reg_k3", "regresses_nonlinear"] == False  # noqa: E712
        # mixed_ksg nonlinear AP = 0.5, clearly below 0.68 -> regresses.
        assert table.loc["mixed_ksg", "mean_ap_nonlinear"] == pytest.approx(0.5)
        assert table.loc["mixed_ksg", "regresses_nonlinear"] == True  # noqa: E712
        # copula_mi nonlinear AP = 0.75, clearly above 0.68 -> does not regress.
        assert table.loc["copula_mi", "mean_ap_nonlinear"] == pytest.approx(0.75)
        assert table.loc["copula_mi", "regresses_nonlinear"] == False  # noqa: E712

    def test_missing_incumbent_does_not_crash_and_never_regresses(self) -> None:
        rows = [
            _mirow(estimator="mixed_ksg", dependence="nonlinear", seed=0, average_precision=0.1),
            _mirow(estimator="mixed_ksg", dependence="linear", seed=0, average_precision=0.2),
            _mirow(estimator="copula_mi", dependence="nonlinear", seed=0, average_precision=0.9),
        ]
        df = pd.DataFrame(rows)
        table = mi_winner(df)
        assert not table["regresses_nonlinear"].any()
        assert not table["regresses_nonlinear"].isna().any()

    def test_missing_dependence_class_yields_nan_not_crash(self) -> None:
        rows = [
            _mirow(estimator="mi_reg_k3", dependence="nonlinear", seed=0, average_precision=0.7),
            _mirow(estimator="copula_mi", dependence="mixed", seed=0, average_precision=0.6),
        ]
        df = pd.DataFrame(rows)
        table = mi_winner(df).set_index("estimator")
        assert math.isnan(table.loc["copula_mi", "mean_ap_linear"])
        assert bool(table.loc["copula_mi", "regresses_nonlinear"]) is False
