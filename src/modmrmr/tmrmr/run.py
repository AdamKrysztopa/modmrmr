"""Lag-Aware ModMRMR covariate-lag selection use case.

Exposes the public entry point :func:`run_tmrmr`, which
orchestrates the full Lag-Aware ModMRMR pipeline:

1. Validate inputs.
2. Build the forecast-safe lag candidate domain.
3. Delegate to the greedy selector.
4. Assemble the typed :class:`~modmrmr.models.LagAwareModMRMRResult`.

ModMRMR modifies the redundancy part of mRMR-style greedy selection by using
multiplicative maximum-similarity suppression against already-selected features.
"""

from __future__ import annotations

import numpy as np

from modmrmr.models import (
    LagAwareModMRMRConfig,
    LagAwareModMRMRResult,
)
from modmrmr.tmrmr.domain import (
    build_forecast_safe_lag_domain,
)
from modmrmr.tmrmr.selector import run_greedy_selection


def run_tmrmr(
    *,
    target: np.ndarray,
    covariates: dict[str, np.ndarray],
    config: LagAwareModMRMRConfig,
    random_state: int = 42,
    run_id: str = "run",
) -> LagAwareModMRMRResult:
    """Run Lag-Aware ModMRMR covariate-lag selection.

    Evaluates which lagged covariates are forecast-safe (``lag >=
    forecast_horizon + availability_margin``), nonlinear-informative for
    ``y(t)``, and non-redundant with already-selected covariate lags.

    The method applies multiplicative maximum-similarity suppression
    (ModMRMR) against the *already-selected* feature set only; the full
    candidate pool is never used as the redundancy reference.

    Args:
        target: 1-D target time series.
        covariates: Mapping of covariate name to aligned 1-D time series.
            All series must have the same length as ``target``.
        config: Lag-Aware ModMRMR run configuration including scorer specs,
            forecast horizon, availability margin, and candidate lags.
        random_state: Base random seed for all scorer calls.  Must be an
            ``int``; ``numpy.Generator`` objects are not accepted.
        run_id: Human-readable run identifier used in normalizer fit-scope
            metadata for deterministic replay.

    Returns:
        :class:`~modmrmr.models.LagAwareModMRMRResult`
        containing selected, rejected, and blocked candidates with full
        scorer diagnostics.

    Raises:
        ValueError: If ``covariates`` is empty, ``target`` fails validation,
            or any covariate fails validation.
        TypeError: If ``random_state`` is not an ``int``.
    """
    if not isinstance(random_state, int):
        raise TypeError(f"random_state must be an int, got {type(random_state).__name__!r}")

    # Step 1: Build forecast-safe lag domain.
    legal_candidates, blocked = build_forecast_safe_lag_domain(
        target=target,
        covariates=covariates,
        config=config,
    )

    # Step 2: Run greedy selection on legal candidates.
    selected, rejected = run_greedy_selection(
        legal_candidates=legal_candidates,
        target=target,
        covariates=covariates,
        config=config,
        random_state=random_state,
        run_id=run_id,
    )

    # Step 3: Assemble result.
    n_evaluated = len(selected) + len(rejected)
    if n_evaluated != len(legal_candidates):
        raise RuntimeError(
            "Lag-Aware ModMRMR selector must account for every legal candidate exactly once. "
            f"Expected {len(legal_candidates)} legal candidates, got {n_evaluated} "
            "selected+rejected entries."
        )

    th_spec = config.target_history_scorer

    return LagAwareModMRMRResult(
        config=config,
        selected=selected,
        rejected=rejected,
        blocked=blocked,
        n_candidates_evaluated=n_evaluated,
        n_candidates_blocked=len(blocked),
        relevance_scorer_spec=config.relevance_scorer,
        redundancy_scorer_spec=config.redundancy_scorer,
        target_history_scorer_spec=th_spec,
        notes=_build_notes(legal_candidates=legal_candidates, blocked=blocked, config=config),
    )


run_lag_aware_mod_mrmr = run_tmrmr


def _build_notes(
    *,
    legal_candidates: list,
    blocked: list,
    config: LagAwareModMRMRConfig,
) -> list[str]:
    """Build human-readable notes for the result.

    Args:
        legal_candidates: Legal lag candidates.
        blocked: Blocked lag candidates.
        config: Run configuration.

    Returns:
        List of note strings for the result payload.
    """
    notes: list[str] = []

    if blocked:
        notes.append(
            f"{len(blocked)} candidate(s) blocked: lag < "
            f"forecast_horizon={config.forecast_horizon} + "
            f"availability_margin={config.availability_margin}."
        )

    if config.target_history_scorer is None:
        notes.append("Target-history novelty penalty is disabled (target_lags not set).")

    notes.append(
        "Multiplicative maximum-similarity suppression against the "
        "already-selected feature set only."
    )

    return notes
