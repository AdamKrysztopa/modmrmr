"""Phase 0 tests for the Lag-Aware ModMRMR frozen domain contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from modmrmr.models import (
    BlockedLagAwareFeature,
    ForecastSafeLagCandidate,
    LagAwareModMRMRConfig,
    LagAwareModMRMRResult,
    PairwiseScorerSpec,
    RejectedLagAwareFeature,
    ScorerDiagnostics,
    SelectedLagAwareFeature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_scorer_spec(
    name: str = "catt_knn_mi",
    backend: str = "ksg",
    normalization: str = "rank_percentile",
    significance: str = "none",
) -> PairwiseScorerSpec:
    return PairwiseScorerSpec(
        name=name,
        backend=backend,
        normalization=normalization,  # type: ignore[arg-type]
        significance_method=significance,  # type: ignore[arg-type]
    )


def _minimal_diagnostics(
    raw_value: float = 0.25,
    normalized_value: float = 0.6,
) -> ScorerDiagnostics:
    return ScorerDiagnostics(
        raw_value=raw_value,
        normalized_value=normalized_value,
        n_pairs=100,
        normalization="rank_percentile",
        significance_method="none",
    )


def _minimal_config(
    forecast_horizon: int = 3,
    availability_margin: int = 0,
    with_target_history: bool = False,
) -> LagAwareModMRMRConfig:
    spec = _minimal_scorer_spec()
    kwargs: dict = dict(
        forecast_horizon=forecast_horizon,
        availability_margin=availability_margin,
        relevance_scorer=spec,
        redundancy_scorer=spec,
    )
    if with_target_history:
        kwargs["target_lags"] = [4, 5]
        kwargs["target_history_scorer"] = spec
    return LagAwareModMRMRConfig(**kwargs)


def _minimal_selected(
    selection_rank: int = 1,
    legality_reason: str = "legal",
    is_known_future: bool = False,
    known_future_provenance=None,
) -> SelectedLagAwareFeature:
    return SelectedLagAwareFeature(
        covariate_name="x1",
        lag=4,
        is_known_future=is_known_future,
        known_future_provenance=known_future_provenance,
        legality_reason=legality_reason,  # type: ignore[arg-type]
        feature_name="x_x1_lag4",
        relevance=0.7,
        max_redundancy=0.1,
        target_history_redundancy=0.0,
        final_score=0.63,
        selection_rank=selection_rank,
        relevance_scorer_name="catt_knn_mi",
        redundancy_scorer_name="catt_knn_mi",
        normalization_strategy="rank_percentile",
    )


def _minimal_rejected() -> RejectedLagAwareFeature:
    return RejectedLagAwareFeature(
        covariate_name="x2",
        lag=5,
        legality_reason="legal",
        feature_name="x_x2_lag5",
        relevance=0.1,
        max_redundancy=0.0,
        target_history_redundancy=0.0,
        final_score=0.1,
        rejection_reason="below_relevance_floor",
    )


def _minimal_blocked() -> BlockedLagAwareFeature:
    return BlockedLagAwareFeature(
        covariate_name="x3",
        lag=1,
        legality_reason="blocked_lag_too_small",
        feature_name="x_x3_lag1",
        block_reason="lag=1 < forecast_horizon=3 + availability_margin=0",
    )


def _minimal_result(
    n_selected: int = 1,
    n_rejected: int = 1,
    n_blocked: int = 1,
    with_target_history: bool = False,
) -> LagAwareModMRMRResult:
    config = _minimal_config(with_target_history=with_target_history)
    spec = _minimal_scorer_spec()
    selected = [_minimal_selected(selection_rank=i + 1) for i in range(n_selected)]
    rejected = [_minimal_rejected() for _ in range(n_rejected)]
    blocked = [_minimal_blocked() for _ in range(n_blocked)]
    kwargs: dict = dict(
        config=config,
        selected=selected,
        rejected=rejected,
        blocked=blocked,
        n_candidates_evaluated=n_selected + n_rejected,
        n_candidates_blocked=n_blocked,
        relevance_scorer_spec=spec,
        redundancy_scorer_spec=spec,
    )
    if with_target_history:
        kwargs["target_history_scorer_spec"] = spec
    return LagAwareModMRMRResult(**kwargs)


# ---------------------------------------------------------------------------
# ScorerDiagnostics
# ---------------------------------------------------------------------------


class TestScorerDiagnostics:
    def test_happy_path(self) -> None:
        diag = _minimal_diagnostics()
        assert diag.raw_value == 0.25
        assert diag.normalized_value == 0.6
        assert diag.n_pairs == 100
        assert diag.p_value is None
        assert diag.adjusted_p_value is None
        assert diag.bands == []
        assert diag.warnings == []

    def test_with_p_values(self) -> None:
        diag = ScorerDiagnostics(
            raw_value=0.3,
            normalized_value=0.7,
            n_pairs=50,
            normalization="surrogate_effect_clip",
            significance_method="bh_fdr_adjustment",
            p_value=0.04,
            adjusted_p_value=0.08,
        )
        assert diag.p_value == pytest.approx(0.04)
        assert diag.adjusted_p_value == pytest.approx(0.08)

    def test_normalized_value_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScorerDiagnostics(
                raw_value=-0.1,
                normalized_value=-0.01,
                n_pairs=100,
                normalization="none",
                significance_method="none",
            )

    def test_normalized_value_zero_allowed(self) -> None:
        diag = ScorerDiagnostics(
            raw_value=0.0,
            normalized_value=0.0,
            n_pairs=10,
            normalization="none",
            significance_method="none",
        )
        assert diag.normalized_value == 0.0

    def test_n_pairs_lt_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScorerDiagnostics(
                raw_value=0.1,
                normalized_value=0.1,
                n_pairs=0,
                normalization="none",
                significance_method="none",
            )

    def test_immutable(self) -> None:
        diag = _minimal_diagnostics()
        with pytest.raises((ValidationError, TypeError)):
            diag.raw_value = 99.9

    @pytest.mark.parametrize(
        "strategy",
        ["rank_percentile", "surrogate_effect_clip", "nmi_min_entropy", "nmi_mean_entropy", "none"],
    )
    def test_all_normalization_strategies(self, strategy: str) -> None:
        diag = ScorerDiagnostics(
            raw_value=0.5,
            normalized_value=0.5,
            n_pairs=20,
            normalization=strategy,  # type: ignore[arg-type]
            significance_method="none",
        )
        assert diag.normalization == strategy

    @pytest.mark.parametrize("method", ["none", "upper_tail_mi_surrogate", "bh_fdr_adjustment"])
    def test_all_significance_methods(self, method: str) -> None:
        diag = ScorerDiagnostics(
            raw_value=0.5,
            normalized_value=0.5,
            n_pairs=20,
            normalization="none",
            significance_method=method,  # type: ignore[arg-type]
        )
        assert diag.significance_method == method


# ---------------------------------------------------------------------------
# PairwiseScorerSpec
# ---------------------------------------------------------------------------


class TestPairwiseScorerSpec:
    def test_happy_path(self) -> None:
        spec = _minimal_scorer_spec()
        assert spec.name == "catt_knn_mi"
        assert spec.backend == "ksg"
        assert spec.normalization == "rank_percentile"
        assert spec.significance_method == "none"
        assert spec.settings == {}

    def test_with_settings(self) -> None:
        spec = PairwiseScorerSpec(
            name="spearman_abs",
            backend="scipy",
            normalization="none",
            significance_method="upper_tail_mi_surrogate",
            settings={"n_surrogates": 99},
        )
        assert spec.settings == {"n_surrogates": 99}

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            PairwiseScorerSpec(
                name="",
                backend="ksg",
                normalization="none",
                significance_method="none",
            )

    def test_whitespace_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            PairwiseScorerSpec(
                name="   ",
                backend="ksg",
                normalization="none",
                significance_method="none",
            )

    def test_empty_backend_raises(self) -> None:
        with pytest.raises(ValidationError):
            PairwiseScorerSpec(
                name="catt_knn_mi",
                backend="",
                normalization="none",
                significance_method="none",
            )

    def test_immutable(self) -> None:
        spec = _minimal_scorer_spec()
        with pytest.raises((ValidationError, TypeError)):
            spec.name = "other"


# ---------------------------------------------------------------------------
# LagAwareModMRMRConfig
# ---------------------------------------------------------------------------


class TestLagAwareModMRMRConfig:
    def test_happy_path(self) -> None:
        cfg = _minimal_config()
        assert cfg.forecast_horizon == 3
        assert cfg.availability_margin == 0
        assert cfg.max_lag == 20
        assert cfg.candidate_lags is None
        assert cfg.target_lags is None
        assert cfg.target_history_scorer is None
        assert cfg.known_future_covariates == {}
        assert cfg.max_selected_features == 10
        assert cfg.relevance_floor == 0.0

    def test_with_target_history(self) -> None:
        cfg = _minimal_config(with_target_history=True)
        assert cfg.target_lags == [4, 5]
        assert cfg.target_history_scorer is not None

    def test_forecast_horizon_zero_raises(self) -> None:
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRConfig(
                forecast_horizon=0,
                relevance_scorer=spec,
                redundancy_scorer=spec,
            )

    def test_target_lags_without_scorer_raises(self) -> None:
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRConfig(
                forecast_horizon=3,
                relevance_scorer=spec,
                redundancy_scorer=spec,
                target_lags=[4, 5],
            )

    def test_target_scorer_without_lags_raises(self) -> None:
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRConfig(
                forecast_horizon=3,
                relevance_scorer=spec,
                redundancy_scorer=spec,
                target_history_scorer=spec,
            )

    def test_empty_candidate_lags_raises(self) -> None:
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRConfig(
                forecast_horizon=3,
                relevance_scorer=spec,
                redundancy_scorer=spec,
                candidate_lags=[],
            )

    def test_candidate_lags_with_zero_raises(self) -> None:
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRConfig(
                forecast_horizon=3,
                relevance_scorer=spec,
                redundancy_scorer=spec,
                candidate_lags=[0, 1, 2],
            )

    def test_known_future_covariates(self) -> None:
        spec = _minimal_scorer_spec()
        cfg = LagAwareModMRMRConfig(
            forecast_horizon=3,
            relevance_scorer=spec,
            redundancy_scorer=spec,
            known_future_covariates={"holiday_flag": "calendar", "setpoint": "contractual"},
        )
        assert cfg.known_future_covariates["holiday_flag"] == "calendar"

    def test_immutable(self) -> None:
        cfg = _minimal_config()
        with pytest.raises((ValidationError, TypeError)):
            cfg.forecast_horizon = 99


# ---------------------------------------------------------------------------
# ForecastSafeLagCandidate
# ---------------------------------------------------------------------------


class TestForecastSafeLagCandidate:
    def test_happy_path_legal(self) -> None:
        c = ForecastSafeLagCandidate(
            covariate_name="x1",
            lag=5,
            is_legal=True,
            legality_reason="legal",
            feature_name="x_x1_lag5",
        )
        assert c.is_legal is True
        assert c.legality_reason == "legal"
        assert c.known_future_provenance is None

    def test_happy_path_blocked(self) -> None:
        c = ForecastSafeLagCandidate(
            covariate_name="x2",
            lag=1,
            is_legal=False,
            legality_reason="blocked_lag_too_small",
            feature_name="x_x2_lag1",
        )
        assert c.is_legal is False

    def test_known_future_legal(self) -> None:
        c = ForecastSafeLagCandidate(
            covariate_name="holiday",
            lag=0,
            is_known_future=True,
            known_future_provenance="calendar",
            is_legal=True,
            legality_reason="legal_known_future",
            feature_name="x_holiday_lag0",
        )
        assert c.known_future_provenance == "calendar"
        assert c.legality_reason == "legal_known_future"

    def test_known_future_true_no_provenance_raises(self) -> None:
        with pytest.raises(ValidationError):
            ForecastSafeLagCandidate(
                covariate_name="x1",
                lag=0,
                is_known_future=True,
                known_future_provenance=None,
                is_legal=True,
                legality_reason="legal_known_future",
                feature_name="x_x1_lag0",
            )

    def test_known_future_false_with_provenance_raises(self) -> None:
        with pytest.raises(ValidationError):
            ForecastSafeLagCandidate(
                covariate_name="x1",
                lag=5,
                is_known_future=False,
                known_future_provenance="calendar",
                is_legal=True,
                legality_reason="legal",
                feature_name="x_x1_lag5",
            )

    @pytest.mark.parametrize(
        "provenance",
        ["calendar", "schedule", "contractual", "forecasted_input"],
    )
    def test_all_provenances(self, provenance: str) -> None:
        c = ForecastSafeLagCandidate(
            covariate_name="kf",
            lag=0,
            is_known_future=True,
            known_future_provenance=provenance,  # type: ignore[arg-type]
            is_legal=True,
            legality_reason="legal_known_future",
            feature_name="x_kf_lag0",
        )
        assert c.known_future_provenance == provenance


# ---------------------------------------------------------------------------
# BlockedLagAwareFeature
# ---------------------------------------------------------------------------


class TestBlockedLagAwareFeature:
    def test_happy_path(self) -> None:
        b = _minimal_blocked()
        assert b.covariate_name == "x3"
        assert b.lag == 1
        assert b.legality_reason == "blocked_lag_too_small"
        assert "lag=1" in b.block_reason

    def test_known_future_blocked(self) -> None:
        b = BlockedLagAwareFeature(
            covariate_name="kf",
            lag=0,
            is_known_future=True,
            known_future_provenance="schedule",
            legality_reason="blocked_known_future",
            feature_name="x_kf_lag0",
            block_reason="realized future observation detected",
        )
        assert b.known_future_provenance == "schedule"

    def test_immutable(self) -> None:
        b = _minimal_blocked()
        with pytest.raises((ValidationError, TypeError)):
            b.block_reason = "other"


# ---------------------------------------------------------------------------
# SelectedLagAwareFeature
# ---------------------------------------------------------------------------


class TestSelectedLagAwareFeature:
    def test_happy_path(self) -> None:
        f = _minimal_selected()
        assert f.selection_rank == 1
        assert f.legality_reason == "legal"
        assert f.relevance == pytest.approx(0.7)
        assert f.max_redundancy == pytest.approx(0.1)
        assert f.target_history_redundancy == pytest.approx(0.0)
        assert f.final_score == pytest.approx(0.63)

    def test_legal_known_future_allowed(self) -> None:
        f = _minimal_selected(
            legality_reason="legal_known_future",
            is_known_future=True,
            known_future_provenance="calendar",
        )
        assert f.legality_reason == "legal_known_future"

    def test_blocked_legality_raises(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_selected(legality_reason="blocked_lag_too_small")

    def test_blocked_known_future_legality_raises(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_selected(legality_reason="blocked_known_future")

    def test_known_future_without_provenance_raises(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_selected(is_known_future=True, known_future_provenance=None)

    def test_not_known_future_with_provenance_raises(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_selected(is_known_future=False, known_future_provenance="calendar")

    def test_negative_relevance_raises(self) -> None:
        with pytest.raises(ValidationError):
            SelectedLagAwareFeature(
                covariate_name="x1",
                lag=4,
                legality_reason="legal",
                feature_name="x_x1_lag4",
                relevance=-0.1,
                max_redundancy=0.0,
                target_history_redundancy=0.0,
                final_score=0.0,
                selection_rank=1,
                relevance_scorer_name="catt_knn_mi",
                redundancy_scorer_name="catt_knn_mi",
                normalization_strategy="rank_percentile",
            )

    def test_max_redundancy_gt_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            SelectedLagAwareFeature(
                covariate_name="x1",
                lag=4,
                legality_reason="legal",
                feature_name="x_x1_lag4",
                relevance=0.5,
                max_redundancy=1.1,
                target_history_redundancy=0.0,
                final_score=0.0,
                selection_rank=1,
                relevance_scorer_name="catt_knn_mi",
                redundancy_scorer_name="catt_knn_mi",
                normalization_strategy="rank_percentile",
            )

    def test_selection_rank_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            SelectedLagAwareFeature(
                covariate_name="x1",
                lag=4,
                legality_reason="legal",
                feature_name="x_x1_lag4",
                relevance=0.5,
                max_redundancy=0.0,
                target_history_redundancy=0.0,
                final_score=0.5,
                selection_rank=0,
                relevance_scorer_name="catt_knn_mi",
                redundancy_scorer_name="catt_knn_mi",
                normalization_strategy="rank_percentile",
            )

    def test_with_diagnostics(self) -> None:
        diag = _minimal_diagnostics()
        f = SelectedLagAwareFeature(
            covariate_name="x1",
            lag=4,
            legality_reason="legal",
            feature_name="x_x1_lag4",
            relevance=0.7,
            max_redundancy=0.0,
            target_history_redundancy=0.0,
            final_score=0.7,
            selection_rank=1,
            relevance_scorer_name="catt_knn_mi",
            redundancy_scorer_name="catt_knn_mi",
            normalization_strategy="rank_percentile",
            relevance_diagnostics=diag,
        )
        assert f.relevance_diagnostics is not None
        assert f.relevance_diagnostics.normalized_value == pytest.approx(0.6)

    def test_immutable(self) -> None:
        f = _minimal_selected()
        with pytest.raises((ValidationError, TypeError)):
            f.selection_rank = 99


# ---------------------------------------------------------------------------
# RejectedLagAwareFeature
# ---------------------------------------------------------------------------


class TestRejectedLagAwareFeature:
    def test_happy_path(self) -> None:
        r = _minimal_rejected()
        assert r.covariate_name == "x2"
        assert r.rejection_reason == "below_relevance_floor"
        assert r.relevance_diagnostics is None

    @pytest.mark.parametrize(
        "reason",
        [
            "below_relevance_floor",
            "zero_final_score",
            "max_features_reached",
            "dominated_by_selected",
        ],
    )
    def test_all_rejection_reasons(self, reason: str) -> None:
        r = RejectedLagAwareFeature(
            covariate_name="x2",
            lag=5,
            legality_reason="legal",
            feature_name="x_x2_lag5",
            relevance=0.1,
            max_redundancy=0.0,
            target_history_redundancy=0.0,
            final_score=0.1,
            rejection_reason=reason,  # type: ignore[arg-type]
        )
        assert r.rejection_reason == reason

    def test_known_future_without_provenance_raises(self) -> None:
        with pytest.raises(ValidationError):
            RejectedLagAwareFeature(
                covariate_name="kf",
                lag=0,
                is_known_future=True,
                known_future_provenance=None,
                legality_reason="legal_known_future",
                feature_name="x_kf_lag0",
                relevance=0.3,
                max_redundancy=0.0,
                target_history_redundancy=0.0,
                final_score=0.3,
                rejection_reason="max_features_reached",
            )


# ---------------------------------------------------------------------------
# LagAwareModMRMRResult
# ---------------------------------------------------------------------------


class TestLagAwareModMRMRResult:
    def test_happy_path(self) -> None:
        result = _minimal_result()
        assert len(result.selected) == 1
        assert len(result.rejected) == 1
        assert len(result.blocked) == 1
        assert result.n_candidates_evaluated == 2
        assert result.n_candidates_blocked == 1
        assert result.target_history_scorer_spec is None

    def test_empty_lists(self) -> None:
        spec = _minimal_scorer_spec()
        result = LagAwareModMRMRResult(
            config=_minimal_config(),
            selected=[],
            rejected=[],
            blocked=[],
            n_candidates_evaluated=0,
            n_candidates_blocked=0,
            relevance_scorer_spec=spec,
            redundancy_scorer_spec=spec,
        )
        assert result.selected == []
        assert result.notes == []

    def test_with_target_history(self) -> None:
        result = _minimal_result(with_target_history=True)
        assert result.target_history_scorer_spec is not None

    def test_wrong_rank_order_raises(self) -> None:
        config = _minimal_config()
        spec = _minimal_scorer_spec()
        # First feature has rank=2 instead of 1
        f1 = _minimal_selected(selection_rank=2)
        with pytest.raises(ValidationError):
            LagAwareModMRMRResult(
                config=config,
                selected=[f1],
                rejected=[],
                blocked=[],
                n_candidates_evaluated=1,
                n_candidates_blocked=0,
                relevance_scorer_spec=spec,
                redundancy_scorer_spec=spec,
            )

    def test_rank_gap_raises(self) -> None:
        config = _minimal_config()
        spec = _minimal_scorer_spec()
        f1 = _minimal_selected(selection_rank=1)
        # Second feature should be rank=2 but is rank=3
        f3 = SelectedLagAwareFeature(
            covariate_name="x2",
            lag=5,
            legality_reason="legal",
            feature_name="x_x2_lag5",
            relevance=0.5,
            max_redundancy=0.1,
            target_history_redundancy=0.0,
            final_score=0.45,
            selection_rank=3,
            relevance_scorer_name="catt_knn_mi",
            redundancy_scorer_name="catt_knn_mi",
            normalization_strategy="rank_percentile",
        )
        with pytest.raises(ValidationError):
            LagAwareModMRMRResult(
                config=config,
                selected=[f1, f3],
                rejected=[],
                blocked=[],
                n_candidates_evaluated=2,
                n_candidates_blocked=0,
                relevance_scorer_spec=spec,
                redundancy_scorer_spec=spec,
            )

    def test_n_candidates_blocked_mismatch_raises(self) -> None:
        config = _minimal_config()
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRResult(
                config=config,
                selected=[],
                rejected=[],
                blocked=[_minimal_blocked()],
                n_candidates_evaluated=0,
                n_candidates_blocked=0,  # should be 1
                relevance_scorer_spec=spec,
                redundancy_scorer_spec=spec,
            )

    def test_n_candidates_evaluated_too_low_raises(self) -> None:
        config = _minimal_config()
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRResult(
                config=config,
                selected=[_minimal_selected(selection_rank=1)],
                rejected=[_minimal_rejected()],
                blocked=[],
                n_candidates_evaluated=1,  # should be >= 2
                n_candidates_blocked=0,
                relevance_scorer_spec=spec,
                redundancy_scorer_spec=spec,
            )

    def test_n_candidates_evaluated_too_high_raises(self) -> None:
        config = _minimal_config()
        spec = _minimal_scorer_spec()
        with pytest.raises(ValidationError):
            LagAwareModMRMRResult(
                config=config,
                selected=[_minimal_selected(selection_rank=1)],
                rejected=[_minimal_rejected()],
                blocked=[],
                n_candidates_evaluated=3,
                n_candidates_blocked=0,
                relevance_scorer_spec=spec,
                redundancy_scorer_spec=spec,
            )

    def test_target_history_spec_present_config_absent_raises(self) -> None:
        spec = _minimal_scorer_spec()
        config = _minimal_config(with_target_history=False)
        with pytest.raises(ValidationError):
            LagAwareModMRMRResult(
                config=config,
                selected=[],
                rejected=[],
                blocked=[],
                n_candidates_evaluated=0,
                n_candidates_blocked=0,
                relevance_scorer_spec=spec,
                redundancy_scorer_spec=spec,
                target_history_scorer_spec=spec,  # present but config has no scorer
            )

    def test_target_history_spec_absent_config_present_raises(self) -> None:
        spec = _minimal_scorer_spec()
        config = _minimal_config(with_target_history=True)
        with pytest.raises(ValidationError):
            LagAwareModMRMRResult(
                config=config,
                selected=[],
                rejected=[],
                blocked=[],
                n_candidates_evaluated=0,
                n_candidates_blocked=0,
                relevance_scorer_spec=spec,
                redundancy_scorer_spec=spec,
                # target_history_scorer_spec absent but config has it
            )

    def test_json_round_trip(self) -> None:
        result = _minimal_result()
        raw = result.model_dump_json()
        parsed = LagAwareModMRMRResult.model_validate_json(raw)
        assert parsed.n_candidates_evaluated == result.n_candidates_evaluated
        assert parsed.n_candidates_blocked == result.n_candidates_blocked
        assert len(parsed.selected) == len(result.selected)
        assert len(parsed.rejected) == len(result.rejected)
        assert len(parsed.blocked) == len(result.blocked)
        # Verify round-trip is stable
        assert json.loads(parsed.model_dump_json()) == json.loads(raw)

    def test_immutable(self) -> None:
        result = _minimal_result()
        with pytest.raises((ValidationError, TypeError)):
            result.n_candidates_evaluated = 999
