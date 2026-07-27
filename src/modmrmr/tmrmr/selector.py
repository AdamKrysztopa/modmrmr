"""Lag-Aware ModMRMR greedy selector with target-history novelty penalty.

Implements the deterministic greedy ModMRMR selection algorithm:

1. Compute raw relevance scores for all legal candidates.
2. Fit a pool normalizer on the relevance pool.
3. Apply the relevance floor to reject low-relevance candidates.
4. Precompute all pairwise redundancy scores between remaining candidates.
5. Fit a pool normalizer on the redundancy pool.
6. Optionally compute target-history novelty penalties.
7. Run the greedy selection loop:
   - First feature: maximum normalized relevance.
   - Subsequent features: maximum ModMRMR score
     ``relevance * (1 - max_redundancy) * (1 - target_history_redundancy)``
     where ``max_redundancy`` is the maximum similarity against the
     already-selected set only (never the full candidate pool).
8. Deterministic tie-breaking: higher relevance, lower max_redundancy, lower
   target_history_redundancy, covariate_name lexicographic, lag ascending.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from modmrmr.core.scorers import (
    PairwiseDependenceScorer,
    _RawScore,
    build_normalizer,
    build_scorer,
    make_diagnostics,
)
from modmrmr.models import (
    ForecastSafeLagCandidate,
    LagAwareModMRMRConfig,
    RejectedLagAwareFeature,
    RejectionReason,
    ScorerDiagnostics,
    SelectedLagAwareFeature,
)
from modmrmr.tmrmr.domain import (
    build_aligned_pair,
    validate_target_history_lags,
)

_ZERO_SCORE_TOLERANCE = 1e-12

# ---------------------------------------------------------------------------
# Internal working state
# ---------------------------------------------------------------------------


@dataclass
class _CandidateState:
    """Mutable working state for one legal candidate during greedy selection.

    Attributes:
        candidate: The legal lag candidate.
        z_lagged: Aligned lagged-covariate array.
        y_target: Aligned target array.
        raw_relevance: Raw relevance score from scorer.
        norm_relevance: Normalized relevance (set after pool fitting).
        raw_redundancy: Row of raw redundancy scores vs. other candidates.
        norm_redundancy: Row of normalized redundancy scores (set after fitting).
        raw_target_history: Raw scores vs. each valid target-history lag.
        norm_target_history: Normalized target-history scores (set after fitting).
        relevance_diagnostics: Full diagnostics; assembled after normalization.
    """

    candidate: ForecastSafeLagCandidate
    z_lagged: np.ndarray
    y_target: np.ndarray
    raw_relevance: _RawScore | None = None
    norm_relevance: float = 0.0
    raw_redundancy: dict[str, _RawScore] = field(default_factory=dict)
    norm_redundancy: dict[str, float] = field(default_factory=dict)
    raw_target_history: dict[int, _RawScore] = field(default_factory=dict)
    norm_target_history: dict[int, float] = field(default_factory=dict)
    relevance_diagnostics: ScorerDiagnostics | None = None

    @property
    def key(self) -> str:
        """Unique string key for this candidate."""
        return f"{self.candidate.covariate_name}__lag{self.candidate.lag}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_candidate_states(
    legal_candidates: list[ForecastSafeLagCandidate],
    target: np.ndarray,
    covariates: dict[str, np.ndarray],
) -> list[_CandidateState]:
    """Build aligned series for every legal candidate.

    Args:
        legal_candidates: Legal lag candidates from the domain builder.
        target: 1-D target series.
        covariates: Covariate name → series mapping.

    Returns:
        List of :class:`_CandidateState` with aligned series populated.
    """
    states: list[_CandidateState] = []
    for cand in legal_candidates:
        z, y = build_aligned_pair(target, covariates[cand.covariate_name], lag=cand.lag)
        states.append(_CandidateState(candidate=cand, z_lagged=z, y_target=y))
    return states


def _score_relevance(
    states: list[_CandidateState],
    scorer: PairwiseDependenceScorer,
    *,
    random_state: int,
) -> None:
    """Compute and store raw relevance scores in-place.

    Args:
        states: Candidate states to score.
        scorer: Relevance scorer implementation.
        random_state: Base random seed; incremented per candidate for
            determinism.
    """
    for idx, st in enumerate(states):
        st.raw_relevance = scorer.score_pair(
            st.z_lagged, st.y_target, random_state=random_state + idx
        )


def _normalize_relevance(
    states: list[_CandidateState],
    config: LagAwareModMRMRConfig,
    *,
    run_id: str,
) -> None:
    """Fit pool normalizer on relevance scores and update states in-place.

    Args:
        states: Candidate states with raw_relevance populated.
        config: Run configuration for normalization strategy.
        run_id: Human-readable run identifier for fit-scope tracking.
    """
    strategy = config.relevance_scorer.normalization
    pool = [st.raw_relevance.raw_value for st in states if st.raw_relevance is not None]
    normalizer = build_normalizer(strategy, fit_scope_id=f"relevance_{run_id}")
    if pool:
        normalizer.fit(pool)

    for st in states:
        if st.raw_relevance is None:
            st.norm_relevance = 0.0
            continue
        raw = st.raw_relevance
        norm = normalizer.transform(raw.raw_value)
        norm = max(norm, 0.0)
        st.norm_relevance = norm
        st.relevance_diagnostics = make_diagnostics(
            raw,
            normalized_value=norm,
            normalization=strategy,
            significance_method=config.relevance_scorer.significance_method,
        )


def _apply_relevance_floor(
    states: list[_CandidateState],
    config: LagAwareModMRMRConfig,
) -> tuple[list[_CandidateState], list[RejectedLagAwareFeature]]:
    """Split candidates into eligible and floor-rejected.

    Args:
        states: All candidate states with normalized relevance set.
        config: Run configuration for relevance_floor.

    Returns:
        Tuple of ``(eligible_states, floor_rejected)``.
    """
    eligible: list[_CandidateState] = []
    rejected: list[RejectedLagAwareFeature] = []

    for st in states:
        if st.norm_relevance <= config.relevance_floor:
            rejected.append(
                RejectedLagAwareFeature(
                    covariate_name=st.candidate.covariate_name,
                    lag=st.candidate.lag,
                    is_known_future=st.candidate.is_known_future,
                    known_future_provenance=st.candidate.known_future_provenance,
                    legality_reason=st.candidate.legality_reason,
                    feature_name=st.candidate.feature_name,
                    relevance=st.norm_relevance,
                    max_redundancy=0.0,
                    target_history_redundancy=0.0,
                    final_score=0.0,
                    rejection_reason="below_relevance_floor",
                    relevance_diagnostics=st.relevance_diagnostics,
                )
            )
        else:
            eligible.append(st)

    return eligible, rejected


# Normalization strategies whose transform does not depend on the fitted pool.
# Only these license lazy redundancy scoring: under any pool-relative strategy,
# computing fewer pairs changes the pool and therefore every normalized value.
# Adding a strategy here is a correctness claim and needs a proof, not a guess.
_POOL_INDEPENDENT_NORMALIZATIONS = frozenset({"none"})


def _redundancy_is_pool_independent(config: LagAwareModMRMRConfig) -> bool:
    """True when redundancy normalization does not depend on the full pool.

    ``rank_percentile`` maps a value to its position within the fitted pool, so
    scoring fewer pairs shifts every result. ``none`` is pass-through and is
    therefore unaffected by which pairs were scored.
    """
    return config.redundancy_scorer.normalization in _POOL_INDEPENDENT_NORMALIZATIONS


def _score_pairwise_redundancy(
    states: list[_CandidateState],
    scorer: PairwiseDependenceScorer,
    *,
    random_state: int,
) -> None:
    """Compute pairwise redundancy scores between all candidate pairs.

    Scores are stored in ``state.raw_redundancy[other.key]``.

    Args:
        states: Eligible candidate states.
        scorer: Redundancy scorer implementation.
        random_state: Base random seed.
    """
    symmetric = getattr(scorer, "symmetric", False)
    n_states = len(states)
    for i, st_a in enumerate(states):
        for j in range(i + 1, n_states):
            st_b = states[j]
            # Clip to common window: z_lagged arrays have length n-lag, which
            # differs across candidates built at different lags.
            min_len = min(len(st_a.z_lagged), len(st_b.z_lagged))
            forward = scorer.score_pair(
                st_a.z_lagged[:min_len],
                st_b.z_lagged[:min_len],
                random_state=random_state + i * n_states + j,
            )
            st_a.raw_redundancy[st_b.key] = forward
            # Both directions are always stored, so the normalization pool at
            # _normalize_redundancy keeps the same multiset and rank_percentile
            # is unchanged. Only the computation is halved.
            st_b.raw_redundancy[st_a.key] = (
                forward
                if symmetric
                else scorer.score_pair(
                    st_b.z_lagged[:min_len],
                    st_a.z_lagged[:min_len],
                    random_state=random_state + j * n_states + i,
                )
            )


def _normalize_redundancy(
    states: list[_CandidateState],
    config: LagAwareModMRMRConfig,
    *,
    run_id: str,
) -> None:
    """Fit pool normalizer on pairwise redundancy scores and update states.

    Args:
        states: Candidate states with raw_redundancy populated.
        config: Run configuration for redundancy normalization strategy.
        run_id: Human-readable run identifier.
    """
    strategy = config.redundancy_scorer.normalization
    pool: list[float] = []
    for st in states:
        pool.extend(r.raw_value for r in st.raw_redundancy.values())

    normalizer = build_normalizer(strategy, fit_scope_id=f"redundancy_{run_id}")
    if pool:
        normalizer.fit(pool)

    for st in states:
        for key, raw in st.raw_redundancy.items():
            norm = normalizer.transform(raw.raw_value)
            st.norm_redundancy[key] = min(max(norm, 0.0), 1.0)


def _score_target_history(
    states: list[_CandidateState],
    target: np.ndarray,
    config: LagAwareModMRMRConfig,
    th_scorer: PairwiseDependenceScorer,
    *,
    valid_target_lags: list[int],
    random_state: int,
) -> None:
    """Compute target-history novelty scores for all candidates.

    For each candidate and each valid target-history lag, compute the raw
    similarity between the candidate's lagged series and the target-history
    series ``y(t - l)``.

    Args:
        states: Eligible candidate states.
        target: 1-D target series.
        config: Run configuration.
        th_scorer: Target-history scorer implementation.
        valid_target_lags: Forecast-safe target-history lags.
        random_state: Base random seed.
    """
    for i, st in enumerate(states):
        for j, th_lag in enumerate(valid_target_lags):
            n = len(target)
            z = st.z_lagged
            y_th = target[: n - th_lag]
            # Align both to the same common window.
            if st.candidate.lag >= th_lag:
                z_aligned = z
                y_aligned = y_th[: len(z)]
            else:
                z_aligned = z[th_lag - st.candidate.lag :]
                y_aligned = y_th[: len(z_aligned)]

            min_len = min(len(z_aligned), len(y_aligned))
            if min_len < 10:
                st.raw_target_history[th_lag] = _RawScore(
                    raw_value=0.0,
                    n_pairs=min_len,
                    warnings=["Too few pairs for target-history scoring."],
                )
                continue

            seed = random_state + i * len(valid_target_lags) + j
            raw = th_scorer.score_pair(z_aligned[:min_len], y_aligned[:min_len], random_state=seed)
            st.raw_target_history[th_lag] = raw


def _normalize_target_history(
    states: list[_CandidateState],
    config: LagAwareModMRMRConfig,
    *,
    run_id: str,
) -> None:
    """Fit pool normalizer on target-history scores and update states.

    Args:
        states: Candidate states with raw_target_history populated.
        config: Run configuration.
        run_id: Human-readable run identifier.
    """
    if config.target_history_scorer is None:
        return

    strategy = config.target_history_scorer.normalization
    pool: list[float] = []
    for st in states:
        pool.extend(r.raw_value for r in st.raw_target_history.values())

    normalizer = build_normalizer(strategy, fit_scope_id=f"target_history_{run_id}")
    if pool:
        normalizer.fit(pool)

    for st in states:
        for th_lag, raw in st.raw_target_history.items():
            norm = normalizer.transform(raw.raw_value)
            st.norm_target_history[th_lag] = min(max(norm, 0.0), 1.0)


def _compute_final_score(
    st: _CandidateState,
    *,
    selected_keys: list[str],
) -> tuple[float, float, float]:
    """Compute the ModMRMR score and penalty components for one candidate.

    Args:
        st: Candidate state with normalized scores populated.
        selected_keys: Keys of already-selected candidates.

    Returns:
        Tuple of ``(final_score, max_redundancy, target_history_redundancy)``.
    """
    max_red = 0.0
    if selected_keys:
        available = [st.norm_redundancy.get(k, 0.0) for k in selected_keys]
        if available:
            max_red = max(available)
    max_red = min(max_red, 1.0)

    th_red = 0.0
    if st.norm_target_history:
        th_red = max(st.norm_target_history.values())
    th_red = min(max(th_red, 0.0), 1.0)

    score = st.norm_relevance * (1.0 - max_red) * (1.0 - th_red)
    score = max(score, 0.0)
    return score, max_red, th_red


def _tie_break_key(
    st: _CandidateState,
    score: float,
) -> tuple[float, float, float, float, str, int]:
    """Deterministic sort key for tie-breaking among candidates.

    Ordering: highest score first, then highest relevance, then lowest
    max_redundancy (computed against no selected, since tie-breaking order
    is used before selection commits), then lowest target-history redundancy,
    then covariate name, then lag ascending.

    Args:
        st: Candidate state.
        score: Precomputed final score.

    Returns:
        Sort key tuple (negate for max ordering where needed).
    """
    th_red = max(st.norm_target_history.values()) if st.norm_target_history else 0.0
    return (
        -score,
        -st.norm_relevance,
        0.0,  # max_redundancy placeholder; computed against selected set
        th_red,
        st.candidate.covariate_name,
        st.candidate.lag,
    )


# ---------------------------------------------------------------------------
# Public selector entry point
# ---------------------------------------------------------------------------


def run_greedy_selection(
    *,
    legal_candidates: list[ForecastSafeLagCandidate],
    target: np.ndarray,
    covariates: dict[str, np.ndarray],
    config: LagAwareModMRMRConfig,
    random_state: int = 42,
    run_id: str = "default",
) -> tuple[list[SelectedLagAwareFeature], list[RejectedLagAwareFeature]]:
    """Run the deterministic Lag-Aware ModMRMR greedy selection.

    Algorithm:

    1. Build aligned series for every legal candidate.
    2. Compute and normalize relevance scores.
    3. Apply relevance floor.
    4. Precompute pairwise redundancy matrix; normalize.
    5. Compute and normalize target-history novelty penalties (when configured).
    6. Greedy loop: select the highest-scoring candidate, update the selected
       set, repeat until ``max_selected_features`` or all exhausted.

    Args:
        legal_candidates: Legal lag candidates from the domain builder.
        target: 1-D target time series.
        covariates: Covariate name → 1-D series mapping.
        config: Lag-Aware ModMRMR run configuration.
        random_state: Base random seed for all scorer calls.
        run_id: Human-readable run identifier used in normalizer fit-scope
            metadata.

    Returns:
        Tuple of ``(selected, rejected)`` feature lists.
    """
    if not legal_candidates:
        return [], []

    rel_scorer = build_scorer(config.relevance_scorer)
    red_scorer = build_scorer(config.redundancy_scorer)
    th_scorer = (
        build_scorer(config.target_history_scorer)
        if config.target_history_scorer is not None
        else None
    )

    # Step 1: build aligned series.
    states = _build_candidate_states(legal_candidates, target, covariates)

    # Step 2: relevance scoring + normalization.
    _score_relevance(states, rel_scorer, random_state=random_state)
    _normalize_relevance(states, config, run_id=run_id)

    # Step 3: relevance floor.
    eligible, floor_rejected = _apply_relevance_floor(states, config)

    if not eligible:
        return [], floor_rejected

    # Step 4: pairwise redundancy.
    _score_pairwise_redundancy(eligible, red_scorer, random_state=random_state + 10000)
    _normalize_redundancy(eligible, config, run_id=run_id)

    # Step 5: target-history novelty penalty.
    valid_th_lags, _ = validate_target_history_lags(config=config)
    if th_scorer is not None and valid_th_lags:
        _score_target_history(
            eligible,
            target,
            config,
            th_scorer,
            valid_target_lags=valid_th_lags,
            random_state=random_state + 20000,
        )
        _normalize_target_history(eligible, config, run_id=run_id)

    # Step 6: greedy loop.
    selected_features: list[SelectedLagAwareFeature] = []
    rejected_features: list[RejectedLagAwareFeature] = list(floor_rejected)
    remaining: list[_CandidateState] = list(eligible)
    selected_keys: list[str] = []

    while remaining and len(selected_features) < config.max_selected_features:
        # Recompute scores for all remaining candidates against current selected.
        scored: list[tuple[float, float, float, _CandidateState]] = []
        for st in remaining:
            score, max_red, th_red = _compute_final_score(st, selected_keys=selected_keys)
            scored.append((score, max_red, th_red, st))

        # Tie-breaking sort: highest score, highest relevance, lowest max_red,
        # lowest th_red, covariate name, lag.
        scored.sort(
            key=lambda t: (
                -t[0],
                -t[3].norm_relevance,
                t[1],
                t[2],
                t[3].candidate.covariate_name,
                t[3].candidate.lag,
            )
        )

        best_score, best_max_red, best_th_red, best_st = scored[0]

        if best_score <= _ZERO_SCORE_TOLERANCE:
            # All remaining candidates have zero score; reject them.
            for _sc, mr, tr, st in scored:
                rejected_features.append(
                    _make_rejected(
                        st,
                        max_redundancy=mr,
                        target_history_redundancy=tr,
                        final_score=0.0,
                        rejection_reason="zero_final_score",
                    )
                )
            remaining = []
            break

        # Select best candidate.
        rank = len(selected_features) + 1
        selected_keys.append(best_st.key)

        # Assemble redundancy diagnostics for the selected candidate.
        red_diag: ScorerDiagnostics | None = None
        if selected_keys and len(selected_keys) > 1:
            # Use the closest-selected redundancy pair as primary diagnostics.
            max_key = max(
                (k for k in selected_keys[:-1]),
                key=lambda k: best_st.norm_redundancy.get(k, 0.0),
                default=None,
            )
            if max_key:
                raw_red = best_st.raw_redundancy.get(max_key)
                if raw_red is not None:
                    red_diag = make_diagnostics(
                        raw_red,
                        normalized_value=best_st.norm_redundancy.get(max_key, 0.0),
                        normalization=config.redundancy_scorer.normalization,
                        significance_method=config.redundancy_scorer.significance_method,
                    )

        # Assemble target-history diagnostics.
        th_diag: ScorerDiagnostics | None = None
        if th_scorer is not None and best_st.raw_target_history:
            # Use the highest-score target-history lag for diagnostics.
            max_th_lag = max(
                best_st.norm_target_history,
                key=lambda k: best_st.norm_target_history[k],
                default=None,
            )
            if max_th_lag is not None:
                raw_th = best_st.raw_target_history.get(max_th_lag)
                if raw_th is not None and config.target_history_scorer is not None:
                    th_diag = make_diagnostics(
                        raw_th,
                        normalized_value=best_st.norm_target_history[max_th_lag],
                        normalization=config.target_history_scorer.normalization,
                        significance_method=config.target_history_scorer.significance_method,
                    )

        th_scorer_name: str | None = (
            config.target_history_scorer.name if config.target_history_scorer is not None else None
        )

        selected_features.append(
            SelectedLagAwareFeature(
                covariate_name=best_st.candidate.covariate_name,
                lag=best_st.candidate.lag,
                is_known_future=best_st.candidate.is_known_future,
                known_future_provenance=best_st.candidate.known_future_provenance,
                legality_reason=best_st.candidate.legality_reason,
                feature_name=best_st.candidate.feature_name,
                relevance=best_st.norm_relevance,
                max_redundancy=best_max_red,
                target_history_redundancy=best_th_red,
                final_score=best_score,
                selection_rank=rank,
                relevance_scorer_name=config.relevance_scorer.name,
                redundancy_scorer_name=config.redundancy_scorer.name,
                target_history_scorer_name=th_scorer_name,
                normalization_strategy=config.relevance_scorer.normalization,
                relevance_diagnostics=best_st.relevance_diagnostics,
                redundancy_diagnostics=red_diag,
                target_history_diagnostics=th_diag,
            )
        )

        # Remove selected candidate from remaining.
        remaining = [st for st in remaining if st.key != best_st.key]

    # Any remaining candidates after the loop are rejected.
    for sc, mr, tr, st in _score_remaining(remaining, selected_keys=selected_keys):
        reason: RejectionReason = (
            "max_features_reached"
            if len(selected_features) >= config.max_selected_features
            else "dominated_by_selected"
        )
        rejected_features.append(
            _make_rejected(
                st,
                max_redundancy=mr,
                target_history_redundancy=tr,
                final_score=sc,
                rejection_reason=reason,
            )
        )

    return selected_features, rejected_features


def _score_remaining(
    remaining: list[_CandidateState],
    *,
    selected_keys: list[str],
) -> list[tuple[float, float, float, _CandidateState]]:
    """Compute final scores for all remaining candidates against selected set.

    Args:
        remaining: Candidates not yet selected.
        selected_keys: Keys of selected candidates.

    Returns:
        List of ``(final_score, max_redundancy, th_redundancy, state)`` tuples.
    """
    result: list[tuple[float, float, float, _CandidateState]] = []
    for st in remaining:
        score, max_red, th_red = _compute_final_score(st, selected_keys=selected_keys)
        result.append((score, max_red, th_red, st))
    return result


def _make_rejected(
    st: _CandidateState,
    *,
    max_redundancy: float,
    target_history_redundancy: float,
    final_score: float,
    rejection_reason: RejectionReason,
) -> RejectedLagAwareFeature:
    """Build a :class:`RejectedLagAwareFeature` from a candidate state.

    Args:
        st: Candidate state.
        max_redundancy: Maximum normalized similarity against selected set.
        target_history_redundancy: Maximum normalized target-history similarity.
        final_score: Final ModMRMR score at the point of rejection.
        rejection_reason: Rejection reason classification.

    Returns:
        Frozen :class:`RejectedLagAwareFeature`.
    """
    return RejectedLagAwareFeature(
        covariate_name=st.candidate.covariate_name,
        lag=st.candidate.lag,
        is_known_future=st.candidate.is_known_future,
        known_future_provenance=st.candidate.known_future_provenance,
        legality_reason=st.candidate.legality_reason,
        feature_name=st.candidate.feature_name,
        relevance=st.norm_relevance,
        max_redundancy=max_redundancy,
        target_history_redundancy=target_history_redundancy,
        final_score=max(final_score, 0.0),
        rejection_reason=rejection_reason,
        relevance_diagnostics=st.relevance_diagnostics,
    )
