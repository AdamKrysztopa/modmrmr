"""Phase 1 tests for the Lag-Aware ModMRMR build logic.

Covers the Phase 1 acceptance criteria:
- Off-by-one legality boundaries
- Known-future bypass labelling
- ModMRMR score formula (with and without target-history)
- Greedy selector determinism
- Duplicate covariate suppression
- Lag-neighbour suppression
- Catt-style AMI scorer availability
- Relevance floor rejection
- Similarity clipping to [0, 1]
- Public facade import
- Max-features cap
- n_candidates_evaluated accounting identity
"""

from __future__ import annotations

import numpy as np

from modmrmr import run_tmrmr as run_lag_aware_mod_mrmr
from modmrmr.models import (
    LagAwareModMRMRConfig,
    LagAwareModMRMRResult,
    PairwiseScorerSpec,
)
from modmrmr.tmrmr.domain import build_forecast_safe_lag_domain

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_CANDIDATE_LAGS_SHORT = [1, 2, 3, 4, 5]
_CANDIDATE_LAGS_FULL = [1, 2, 3, 4, 5, 6, 7, 8]


def _scorer_spec(
    name: str = "spearman_abs",
    normalization: str = "rank_percentile",
) -> PairwiseScorerSpec:
    return PairwiseScorerSpec(
        name=name,
        backend="scipy",
        normalization=normalization,  # type: ignore[arg-type]
        significance_method="none",
    )


def _basic_config(
    h: int = 3,
    m: int = 0,
    max_selected: int = 10,
    candidate_lags: list[int] | None = None,
    max_lag: int = 10,
) -> LagAwareModMRMRConfig:
    spec = _scorer_spec()
    return LagAwareModMRMRConfig(
        forecast_horizon=h,
        availability_margin=m,
        relevance_scorer=spec,
        redundancy_scorer=spec,
        max_selected_features=max_selected,
        candidate_lags=candidate_lags,
        max_lag=max_lag,
    )


def _make_ar1(n: int = 200, phi: float = 0.7, seed: int = 42) -> np.ndarray:
    """Generate an AR(1) series y[t] = phi * y[t-1] + noise."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


def _make_white_noise(n: int = 200, seed: int = 99) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _lead_covariate(
    series: np.ndarray,
    *,
    lag: int,
    seed: int,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """Return a leading covariate whose lagged view aligns with the target."""
    rng = np.random.default_rng(seed)
    head = series[lag:] + noise_scale * rng.standard_normal(series.size - lag)
    tail = rng.standard_normal(lag)
    return np.concatenate([head, tail])


# ---------------------------------------------------------------------------
# 1. Off-by-one legality boundary tests
# ---------------------------------------------------------------------------


def test_legality_boundary_h1_m0_lag1_is_first_legal() -> None:
    """h=1, m=0: cutoff=1, so lag=1 is the first legal lag."""
    target = _make_ar1(100)
    cov = _make_white_noise(100)
    config = _basic_config(h=1, m=0, candidate_lags=[1, 2, 3])
    legal, blocked = build_forecast_safe_lag_domain(
        target=target, covariates={"x": cov}, config=config
    )
    legal_lags = {c.lag for c in legal}
    blocked_lags = {c.lag for c in blocked}
    assert 1 in legal_lags
    assert 2 in legal_lags
    assert 3 in legal_lags
    assert len(blocked_lags) == 0


def test_legality_boundary_h1_m1_lag1_is_blocked() -> None:
    """h=1, m=1: cutoff=2, so lag=1 is blocked and lag=2 is the first legal lag."""
    target = _make_ar1(100)
    cov = _make_white_noise(100)
    config = _basic_config(h=1, m=1, candidate_lags=[1, 2, 3])
    legal, blocked = build_forecast_safe_lag_domain(
        target=target, covariates={"x": cov}, config=config
    )
    legal_lags = {c.lag for c in legal}
    blocked_lags = {c.lag for c in blocked}
    assert 1 in blocked_lags
    assert 2 in legal_lags
    assert 3 in legal_lags


def test_legality_boundary_h3_m2_lag4_is_blocked() -> None:
    """h=3, m=2: cutoff=5, so lags 1-4 are blocked and lag=5 is first legal."""
    target = _make_ar1(100)
    cov = _make_white_noise(100)
    config = _basic_config(h=3, m=2, candidate_lags=[1, 2, 3, 4, 5, 6])
    legal, blocked = build_forecast_safe_lag_domain(
        target=target, covariates={"x": cov}, config=config
    )
    legal_lags = {c.lag for c in legal}
    blocked_lags = {c.lag for c in blocked}
    assert blocked_lags == {1, 2, 3, 4}
    assert 5 in legal_lags
    assert 6 in legal_lags


# ---------------------------------------------------------------------------
# 2. Known-future bypass
# ---------------------------------------------------------------------------


def test_known_future_bypass_all_lags_are_legal_known_future() -> None:
    """Known-future covariates bypass the ordinary lag cutoff and are labelled."""
    target = _make_ar1(100)
    cal = _make_white_noise(100, seed=7)
    config = LagAwareModMRMRConfig(
        forecast_horizon=5,
        availability_margin=2,
        candidate_lags=[1, 2, 3],
        known_future_covariates={"calendar_flag": "calendar"},
        relevance_scorer=_scorer_spec(),
        redundancy_scorer=_scorer_spec(),
    )
    legal, blocked = build_forecast_safe_lag_domain(
        target=target,
        covariates={"calendar_flag": cal},
        config=config,
    )
    assert len(blocked) == 0
    for cand in legal:
        assert cand.legality_reason == "legal_known_future"
        assert cand.is_known_future is True
        assert cand.known_future_provenance == "calendar"


# ---------------------------------------------------------------------------
# 3. Score formula
# ---------------------------------------------------------------------------


def test_score_formula_without_target_history() -> None:
    """final_score == relevance * (1 - max_redundancy) when no target-history."""
    target = _make_ar1(200)
    cov_a = _make_ar1(200, seed=1)
    cov_b = _make_ar1(200, seed=2)
    config = _basic_config(h=1, m=0, candidate_lags=[1, 2, 3, 4], max_selected=5)
    result = run_lag_aware_mod_mrmr(
        target=target, covariates={"a": cov_a, "b": cov_b}, config=config
    )
    for feat in result.selected:
        expected = feat.relevance * (1.0 - feat.max_redundancy)
        assert abs(feat.final_score - expected) < 1e-9, (
            f"{feat.feature_name}: expected {expected}, got {feat.final_score}"
        )
    cap_or_dom = ("max_features_reached", "dominated_by_selected", "zero_final_score")
    for feat in result.rejected:
        if feat.rejection_reason in cap_or_dom:
            expected = feat.relevance * (1.0 - feat.max_redundancy)
            assert abs(feat.final_score - expected) < 1e-9


def test_score_formula_with_target_history_three_term_product() -> None:
    """final_score == rel * (1-max_red) * (1-th_red) with target-history enabled."""
    target = _make_ar1(200)
    cov_a = _make_ar1(200, seed=1)
    spec = _scorer_spec()
    config = LagAwareModMRMRConfig(
        forecast_horizon=1,
        availability_margin=0,
        candidate_lags=[2, 3, 4, 5],
        relevance_scorer=spec,
        redundancy_scorer=spec,
        target_lags=[2, 3],
        target_history_scorer=spec,
        max_selected_features=5,
    )
    result = run_lag_aware_mod_mrmr(target=target, covariates={"a": cov_a}, config=config)
    for feat in result.selected:
        expected = (
            feat.relevance * (1.0 - feat.max_redundancy) * (1.0 - feat.target_history_redundancy)
        )
        assert abs(feat.final_score - expected) < 1e-9, (
            f"{feat.feature_name}: expected {expected:.6f}, got {feat.final_score:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_greedy_selector_determinism_same_random_state() -> None:
    """Identical random_state produces identical selections on two runs."""
    target = _make_ar1(200)
    covs = {"a": _make_ar1(200, seed=1), "b": _make_ar1(200, seed=2)}
    config = _basic_config(h=1, m=0, candidate_lags=_CANDIDATE_LAGS_FULL, max_selected=5)
    r1 = run_lag_aware_mod_mrmr(target=target, covariates=covs, config=config, random_state=0)
    r2 = run_lag_aware_mod_mrmr(target=target, covariates=covs, config=config, random_state=0)
    assert [f.feature_name for f in r1.selected] == [f.feature_name for f in r2.selected]
    assert [f.final_score for f in r1.selected] == [f.final_score for f in r2.selected]


# ---------------------------------------------------------------------------
# 5. Duplicate covariate suppression
# ---------------------------------------------------------------------------


def test_duplicate_covariate_suppression_near_duplicate_has_high_redundancy() -> None:
    """Two perfectly identical covariates: the second is redundancy-suppressed.

    Uses normalization='none' so raw Spearman on identical series produces a
    score near 1.0 rather than a rank-relative value.
    """
    target = _make_ar1(200)
    cov = _make_ar1(200, seed=5)
    spec_raw = PairwiseScorerSpec(
        name="spearman_abs",
        backend="scipy",
        normalization="none",
        significance_method="none",
    )
    config = LagAwareModMRMRConfig(
        forecast_horizon=1,
        availability_margin=0,
        candidate_lags=[2, 3, 4, 5],
        relevance_scorer=spec_raw,
        redundancy_scorer=spec_raw,
        max_selected_features=2,
    )
    result = run_lag_aware_mod_mrmr(
        target=target,
        covariates={"sensor_a": cov, "sensor_b": cov.copy()},
        config=config,
    )
    # With raw Spearman, identical series have correlation ~1.0, so max_redundancy > 0.8.
    assert len(result.rejected) > 0
    max_reds = [f.max_redundancy for f in result.rejected]
    assert any(r > 0.8 for r in max_reds), (
        f"Expected at least one high-redundancy rejection with raw scorer, got: {max_reds}"
    )


# ---------------------------------------------------------------------------
# 6. Lag neighbour suppression
# ---------------------------------------------------------------------------


def test_lag_neighbor_suppression_second_selected_has_nonzero_redundancy() -> None:
    """For an AR(1), adjacent lags are correlated; after first selection the
    second selected or rejected lag should carry nonzero max_redundancy."""
    target = _make_ar1(200, seed=42)
    cov = _make_ar1(200, seed=1)
    config = _basic_config(h=1, m=0, candidate_lags=_CANDIDATE_LAGS_FULL, max_selected=2)
    result = run_lag_aware_mod_mrmr(
        target=target,
        covariates={"x": cov},
        config=config,
    )
    # The second selected (or first rejected) must reflect some redundancy penalty.
    later_features = list(result.selected[1:]) + list(result.rejected)
    assert any(f.max_redundancy > 0.0 for f in later_features), (
        "Expected at least one candidate to carry a nonzero redundancy penalty."
    )


# ---------------------------------------------------------------------------
# 7. Catt-style AMI scorer
# ---------------------------------------------------------------------------


def test_catt_knn_mi_scorer_completes_and_returns_nonzero_relevance() -> None:
    """catt_knn_mi scorer runs to completion and returns positive relevance."""
    target = _make_ar1(200)
    cov = _make_ar1(200, seed=3)
    spec = PairwiseScorerSpec(
        name="catt_knn_mi",
        backend="ksg",
        normalization="rank_percentile",
        significance_method="none",
    )
    config = LagAwareModMRMRConfig(
        forecast_horizon=1,
        availability_margin=0,
        candidate_lags=[2, 3, 4],
        relevance_scorer=spec,
        redundancy_scorer=spec,
        max_selected_features=3,
    )
    result = run_lag_aware_mod_mrmr(target=target, covariates={"x": cov}, config=config)
    assert isinstance(result, LagAwareModMRMRResult)
    if result.selected:
        assert result.selected[0].relevance > 0.0


# ---------------------------------------------------------------------------
# 8. Relevance floor rejection
# ---------------------------------------------------------------------------


def test_relevance_floor_rejects_low_relevance_candidates() -> None:
    """Candidates below the relevance_floor are labelled below_relevance_floor."""
    target = _make_ar1(200)
    noise = _make_white_noise(200, seed=10)
    # Use normalization="none" and a high floor so white noise gets rejected.
    spec_none = PairwiseScorerSpec(
        name="spearman_abs",
        backend="scipy",
        normalization="none",
        significance_method="none",
    )
    config = LagAwareModMRMRConfig(
        forecast_horizon=1,
        availability_margin=0,
        candidate_lags=[2, 3, 4, 5],
        relevance_scorer=spec_none,
        redundancy_scorer=spec_none,
        relevance_floor=0.9,
        max_selected_features=5,
    )
    result = run_lag_aware_mod_mrmr(target=target, covariates={"noise": noise}, config=config)
    floor_rejected = [r for r in result.rejected if r.rejection_reason == "below_relevance_floor"]
    assert len(floor_rejected) > 0, (
        "Expected at least one below_relevance_floor rejection for white noise"
    )


# ---------------------------------------------------------------------------
# 9. Similarity clipping
# ---------------------------------------------------------------------------


def test_similarity_values_are_within_unit_interval() -> None:
    """All max_redundancy and target_history_redundancy values are in [0, 1]."""
    target = _make_ar1(200)
    covs = {"a": _make_ar1(200, seed=1), "b": _make_ar1(200, seed=2)}
    config = _basic_config(h=1, m=0, candidate_lags=_CANDIDATE_LAGS_FULL, max_selected=3)
    result = run_lag_aware_mod_mrmr(target=target, covariates=covs, config=config)
    for feat in result.selected + result.rejected:
        assert 0.0 <= feat.max_redundancy <= 1.0, (
            f"max_redundancy out of range: {feat.max_redundancy}"
        )
        assert 0.0 <= feat.target_history_redundancy <= 1.0, (
            f"target_history_redundancy out of range: {feat.target_history_redundancy}"
        )


# ---------------------------------------------------------------------------
# 10. Public facade import
# ---------------------------------------------------------------------------


def test_public_facade_import_returns_lag_aware_mod_mrmr_result() -> None:
    """run_lag_aware_mod_mrmr imported from modmrmr returns LagAwareModMRMRResult."""
    target = _make_ar1(200)
    cov = _make_ar1(200, seed=1)
    config = _basic_config(h=1, m=0, candidate_lags=[2, 3, 4], max_selected=3)
    result = run_lag_aware_mod_mrmr(target=target, covariates={"x": cov}, config=config)
    assert isinstance(result, LagAwareModMRMRResult)


# ---------------------------------------------------------------------------
# 11. Max-features cap
# ---------------------------------------------------------------------------


def test_max_features_cap_selects_exactly_n_and_labels_rest() -> None:
    """When max_selected_features=2 and many legal candidates exist, exactly 2 are selected."""
    target = _make_ar1(200)
    cov_a = _make_ar1(200, seed=1)
    cov_b = _make_ar1(200, seed=2)
    config = _basic_config(h=1, m=0, candidate_lags=_CANDIDATE_LAGS_FULL, max_selected=2)
    result = run_lag_aware_mod_mrmr(
        target=target,
        covariates={"a": cov_a, "b": cov_b},
        config=config,
    )
    assert len(result.selected) == 2
    cap_rejected = [r for r in result.rejected if r.rejection_reason == "max_features_reached"]
    assert len(cap_rejected) > 0, "Expected at least one max_features_reached rejection"


# ---------------------------------------------------------------------------
# 12. n_candidates_evaluated accounting identity
# ---------------------------------------------------------------------------


def test_n_candidates_evaluated_equals_selected_plus_rejected() -> None:
    """n_candidates_evaluated must equal len(selected) + len(rejected)."""
    target = _make_ar1(200)
    covs = {"a": _make_ar1(200, seed=1), "b": _make_ar1(200, seed=2)}
    config = _basic_config(h=1, m=0, candidate_lags=_CANDIDATE_LAGS_FULL)
    result = run_lag_aware_mod_mrmr(target=target, covariates=covs, config=config)
    assert result.n_candidates_evaluated == len(result.selected) + len(result.rejected)


def test_target_history_proxy_zero_score_is_rejected_once_and_counts_match_domain() -> None:
    """Target-history proxy residue should reject once and preserve candidate accounting."""
    target = _make_ar1(260, phi=0.88, seed=41)
    covariates = {
        "leading_signal": _lead_covariate(target, lag=4, seed=42, noise_scale=0.02),
        "target_proxy": target.copy(),
    }
    spec = PairwiseScorerSpec(
        name="spearman_abs",
        backend="scipy",
        normalization="none",
        significance_method="none",
    )
    config = LagAwareModMRMRConfig(
        forecast_horizon=1,
        availability_margin=0,
        candidate_lags=[4],
        relevance_scorer=spec,
        redundancy_scorer=spec,
        target_lags=[4],
        target_history_scorer=spec,
        max_selected_features=2,
    )

    legal, _blocked = build_forecast_safe_lag_domain(
        target=target,
        covariates=covariates,
        config=config,
    )
    result = run_lag_aware_mod_mrmr(
        target=target,
        covariates=covariates,
        config=config,
    )

    assert [feature.feature_name for feature in result.selected] == ["x_leading_signal_lag4"]
    zero_rejected = [
        feature for feature in result.rejected if feature.rejection_reason == "zero_final_score"
    ]
    assert [feature.feature_name for feature in zero_rejected] == ["x_target_proxy_lag4"]
    assert zero_rejected[0].final_score == 0.0
    assert result.n_candidates_evaluated == len(legal)
    assert len({(feature.covariate_name, feature.lag) for feature in result.rejected}) == len(
        result.rejected
    )
