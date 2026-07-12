"""Forecast-safe lag-domain builder for Lag-Aware ModMRMR.

Constructs the legal lag candidate matrix from raw target and covariate
series, enforcing forecast-horizon legality before any scoring occurs.

A covariate lag ``k`` is legal when::

    k >= forecast_horizon + availability_margin   (ordinary measured covariates)

Known-future covariates bypass the ordinary lag cutoff when declared with
valid provenance.  Realized future observations must not be entered as
known-future.

Target-history lags must also satisfy the same cutoff::

    lag >= forecast_horizon + availability_margin
"""

from __future__ import annotations

import numpy as np

from modmrmr._vendor.validation import validate_time_series
from modmrmr.models import (
    BlockedLagAwareFeature,
    ForecastSafeLagCandidate,
    LagAwareModMRMRConfig,
    LagLegalityLabel,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_feature_name(covariate_name: str, lag: int) -> str:
    """Derive a canonical feature identifier from covariate name and lag.

    Args:
        covariate_name: Raw covariate series name.
        lag: Lag offset k.

    Returns:
        Canonical feature identifier (e.g. ``"x_sensor1_lag3"``).
    """
    safe_name = covariate_name.replace(" ", "_").replace("-", "_")
    return f"x_{safe_name}_lag{lag}"


def _get_candidate_lags(config: LagAwareModMRMRConfig) -> list[int]:
    """Return candidate lags from config or derive from ``max_lag``.

    Args:
        config: Lag-Aware ModMRMR run configuration.

    Returns:
        Sorted list of candidate lag values.
    """
    if config.candidate_lags is not None:
        return sorted(config.candidate_lags)
    return list(range(1, config.max_lag + 1))


def _minimum_required_length(config: LagAwareModMRMRConfig) -> int:
    """Return the minimum series length required for the configured lags.

    Args:
        config: Lag-Aware ModMRMR run configuration.

    Returns:
        Minimum required series length (>= max_candidate_lag + 2).
    """
    candidate_lags = _get_candidate_lags(config)
    max_cand = max(candidate_lags) if candidate_lags else 1
    if config.target_lags:
        max_cand = max(max_cand, max(config.target_lags))
    return max_cand + 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_forecast_safe_lag_domain(
    *,
    target: np.ndarray,
    covariates: dict[str, np.ndarray],
    config: LagAwareModMRMRConfig,
) -> tuple[list[ForecastSafeLagCandidate], list[BlockedLagAwareFeature]]:
    """Build the forecast-safe lag candidate domain.

    Evaluates every ``(covariate, lag)`` pair against the forecast-horizon
    legality rule.  Illegal lags are collected as blocked candidates and
    never enter the scoring pool.  Known-future covariates bypass the
    ordinary lag cutoff when explicitly declared with valid provenance.

    Args:
        target: Target time series (1-D NumPy array).
        covariates: Mapping of covariate name to aligned 1-D time series.
        config: Lag-Aware ModMRMR run configuration.

    Returns:
        Tuple of ``(legal_candidates, blocked_candidates)``.

    Raises:
        ValueError: If ``covariates`` is empty, target fails validation, any
            covariate fails validation, or series lengths are inconsistent.
    """
    if len(covariates) == 0:
        raise ValueError("covariates must contain at least one series")

    min_len = _minimum_required_length(config)
    validated_target = validate_time_series(target, min_length=min_len)
    series_len = int(validated_target.size)

    validated_covariates: dict[str, np.ndarray] = {}
    for name, cov in covariates.items():
        validated_cov = validate_time_series(cov, min_length=min_len)
        if validated_cov.size != series_len:
            raise ValueError(
                f"Covariate '{name}' length {validated_cov.size} does not match "
                f"target length {series_len}."
            )
        validated_covariates[name] = validated_cov

    candidate_lags = _get_candidate_lags(config)
    cutoff = config.forecast_horizon + config.availability_margin

    legal: list[ForecastSafeLagCandidate] = []
    blocked: list[BlockedLagAwareFeature] = []

    # Iterate in sorted order for determinism.
    for covariate_name in sorted(covariates):
        is_known_future = covariate_name in config.known_future_covariates
        provenance = config.known_future_covariates.get(covariate_name)

        for lag in candidate_lags:
            feature_name = _make_feature_name(covariate_name, lag)

            if is_known_future:
                legality_reason: LagLegalityLabel = "legal_known_future"
                legal.append(
                    ForecastSafeLagCandidate(
                        covariate_name=covariate_name,
                        lag=lag,
                        is_known_future=True,
                        known_future_provenance=provenance,
                        is_legal=True,
                        legality_reason=legality_reason,
                        feature_name=feature_name,
                    )
                )
            elif lag >= cutoff:
                legal.append(
                    ForecastSafeLagCandidate(
                        covariate_name=covariate_name,
                        lag=lag,
                        is_known_future=False,
                        known_future_provenance=None,
                        is_legal=True,
                        legality_reason="legal",
                        feature_name=feature_name,
                    )
                )
            else:
                blocked.append(
                    BlockedLagAwareFeature(
                        covariate_name=covariate_name,
                        lag=lag,
                        is_known_future=False,
                        known_future_provenance=None,
                        legality_reason="blocked_lag_too_small",
                        feature_name=feature_name,
                        block_reason=(
                            f"lag={lag} < forecast_horizon={config.forecast_horizon}"
                            f" + availability_margin={config.availability_margin}"
                        ),
                    )
                )

    return legal, blocked


def build_aligned_pair(
    target: np.ndarray,
    covariate: np.ndarray,
    *,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned ``(z_lagged, y_target)`` pair for scoring.

    Produces the lagged covariate and aligned target window for scoring one
    ``(covariate, lag)`` pair.  The output arrays are equal-length with no
    missing values.

    ``z_lagged(t) = covariate(t - lag)``

    Args:
        target: 1-D target time series.
        covariate: 1-D covariate time series aligned with target.
        lag: Lag offset ``k`` (must be >= 0).

    Returns:
        Tuple of ``(z_lagged, y_target)`` with equal length ``n - lag``.

    Raises:
        ValueError: If ``lag < 0``, ``lag >= len(target)``, or lengths differ.
    """
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    n = len(target)
    if len(covariate) != n:
        raise ValueError(f"target length {n} and covariate length {len(covariate)} must match")
    if lag >= n:
        raise ValueError(f"lag={lag} >= series length={n}")

    z_lagged = covariate[: n - lag]
    y_target = target[lag:]
    return z_lagged, y_target


def validate_target_history_lags(
    *,
    config: LagAwareModMRMRConfig,
) -> tuple[list[int], list[int]]:
    """Separate valid and blocked target-history lags.

    Target-history lags must satisfy ``lag >= forecast_horizon +
    availability_margin`` to be forecast-safe.

    Args:
        config: Lag-Aware ModMRMR run configuration.

    Returns:
        Tuple of ``(valid_target_lags, blocked_target_lags)``.
    """
    if config.target_lags is None:
        return [], []

    cutoff = config.forecast_horizon + config.availability_margin
    valid = [lag for lag in config.target_lags if lag >= cutoff]
    blocked_lags = [lag for lag in config.target_lags if lag < cutoff]
    return valid, blocked_lags
