"""Pairwise scorer protocol, open registry, and adapters.

Provides the scoring infrastructure for Lag-Aware ModMRMR:

- :class:`PairwiseDependenceScorer` — protocol for all pairwise scorers.
- :class:`_RawScore` — internal result before normalization.
- An open registry (:func:`register_scorer`, :func:`get_scorer`,
  :func:`list_scorers`) so callers can plug in custom scorers without
  modifying this module.
- :func:`build_scorer` — factory from :class:`PairwiseScorerSpec`.
- :func:`as_importance_function` / :func:`as_penalty_matrix` — adapters that
  turn a pairwise scorer into the importance/penalty callables consumed by
  the selection loop.
- :func:`make_diagnostics` — assembles :class:`ScorerDiagnostics` after
  normalization.

This module supplies the scoring primitives that ModMRMR plugs together; the
method-agnostic design means any compliant scorer can be used for relevance,
redundancy, or target-history novelty penalties.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from modmrmr.core.normalize import build_normalizer  # re-exported for convenience
from modmrmr.models import (
    NormalizationStrategy,
    PairwiseScorerSpec,
    ScorerDiagnostics,
    SignificanceMethod,
)

__all__ = [
    "build_normalizer",
    "_RawScore",
    "PairwiseDependenceScorer",
    "_MIN_PAIRS",
    "make_diagnostics",
    "register_scorer",
    "get_scorer",
    "list_scorers",
    "build_scorer",
    "as_importance_function",
    "as_penalty_matrix",
    "fast_pearson_penalty",
]

# ---------------------------------------------------------------------------
# Internal raw-score container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RawScore:
    """Internal result from a scorer before normalization.

    Attributes:
        raw_value: Original scorer output (unbounded non-negative float).
        n_pairs: Number of aligned non-missing sample pairs scored.
        estimator_settings: JSON-serialisable estimator settings.
        bands: Optional reliability bands or clipping notes.
        warnings: Low-sample, convergence, or estimator warnings.
        p_value: Unadjusted p-value; ``None`` when not computed.
    """

    raw_value: float
    n_pairs: int
    estimator_settings: dict[str, Any] = field(default_factory=dict)
    bands: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    p_value: float | None = None


# ---------------------------------------------------------------------------
# Scorer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PairwiseDependenceScorer(Protocol):
    """Protocol for pairwise dependence scorers.

    A scorer takes two aligned 1-D arrays and returns a :class:`_RawScore`
    with the raw dependence value and metadata.  Normalization is applied
    separately by a :class:`_PoolNormalizer` after collecting the full pool.
    """

    name: str

    def score_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,
    ) -> _RawScore:
        """Score one ``(x, y)`` pair.

        Args:
            x: First 1-D array (lagged covariate or feature).
            y: Second 1-D array (target or another feature).
            random_state: Random seed for stochastic components.

        Returns:
            :class:`_RawScore` with raw dependence value and metadata.
        """
        ...


_MIN_PAIRS = 10  # minimum pair count to attempt scoring


# ---------------------------------------------------------------------------
# Diagnostics assembler
# ---------------------------------------------------------------------------


def make_diagnostics(
    raw: _RawScore,
    *,
    normalized_value: float,
    normalization: NormalizationStrategy,
    significance_method: SignificanceMethod,
    adjusted_p_value: float | None = None,
) -> ScorerDiagnostics:
    """Assemble a :class:`ScorerDiagnostics` from a raw score and normalisation.

    Args:
        raw: Raw scorer output (before normalization).
        normalized_value: Transformed value produced by the pool normalizer.
        normalization: Normalization strategy that produced ``normalized_value``.
        significance_method: Significance method used during scoring.
        adjusted_p_value: Multiple-comparisons-adjusted p-value; ``None``
            unless ``significance_method="bh_fdr_adjustment"``.

    Returns:
        Frozen :class:`ScorerDiagnostics` ready for inclusion in result models.
    """
    clipped = max(normalized_value, 0.0)
    return ScorerDiagnostics(
        raw_value=raw.raw_value,
        normalized_value=clipped,
        p_value=raw.p_value,
        adjusted_p_value=adjusted_p_value,
        n_pairs=max(raw.n_pairs, 1),
        estimator_settings=raw.estimator_settings,
        normalization=normalization,
        significance_method=significance_method,
        bands=list(raw.bands),
        warnings=list(raw.warnings),
    )


# ---------------------------------------------------------------------------
# Open scorer registry
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: dict[str, PairwiseDependenceScorer] = {}


def register_scorer(name: str, scorer: PairwiseDependenceScorer) -> None:
    """Register a scorer instance under ``name`` (overwrites any existing)."""
    if not name.strip():
        raise ValueError("scorer name must be non-empty")
    _SCORER_REGISTRY[name] = scorer


def get_scorer(name: str) -> PairwiseDependenceScorer:
    """Return the registered scorer for ``name``."""
    if name not in _SCORER_REGISTRY:
        supported = ", ".join(sorted(_SCORER_REGISTRY))
        raise ValueError(f"Unknown scorer name {name!r}. Supported: {supported}")
    return _SCORER_REGISTRY[name]


def list_scorers() -> list[str]:
    """Return the sorted names of all registered scorers."""
    return sorted(_SCORER_REGISTRY)


def build_scorer(spec: PairwiseScorerSpec) -> PairwiseDependenceScorer:
    """Build a scorer from a spec, honouring an ``n_neighbors`` setting."""
    scorer = get_scorer(spec.name)
    n_neighbors = spec.settings.get("n_neighbors")
    if n_neighbors is not None and hasattr(scorer, "with_neighbors"):
        return scorer.with_neighbors(int(n_neighbors))
    return scorer


def as_importance_function(
    scorer: PairwiseDependenceScorer, *, random_state: int = 42
) -> Callable[..., np.ndarray]:
    """Adapt a pairwise scorer into ``importance_function(X, y) -> per-feature relevance``."""

    def _importance(X: Any, y: Any, **_: Any) -> np.ndarray:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        yv = np.asarray(y, dtype=float).reshape(-1)
        return np.array(
            [
                scorer.score_pair(
                    X_df[c].to_numpy(dtype=float), yv, random_state=random_state
                ).raw_value
                for c in X_df.columns
            ]
        )

    return _importance


def as_penalty_matrix(
    scorer: PairwiseDependenceScorer, *, random_state: int = 42
) -> Callable[..., pd.DataFrame]:
    """Adapt a pairwise scorer into ``penalty_function(X) -> feature x feature redundancy``."""

    def _penalty(X: Any, **_: Any) -> pd.DataFrame:
        if not getattr(scorer, "supports_redundancy", True):
            raise NotImplementedError(
                f"Scorer {getattr(scorer, 'name', '?')!r} is relevance-only and "
                "cannot be used as a pairwise redundancy penalty."
            )
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = list(X_df.columns)
        arrs = {c: X_df[c].to_numpy(dtype=float) for c in cols}
        out = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                out.loc[a, b] = (
                    1.0
                    if i == j
                    else scorer.score_pair(arrs[a], arrs[b], random_state=random_state).raw_value
                )
        return out

    return _penalty


def fast_pearson_penalty(X: pd.DataFrame) -> pd.DataFrame:
    """Vectorized |Pearson| penalty matrix — exact fast path for the ``pearson_abs`` loop.

    Requires fully finite input; the pairwise loop in ``as_penalty_matrix`` remains the
    NaN-tolerant reference path.
    """
    arr = X.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(
            "fast_pearson_penalty requires fully finite input; "
            "use as_penalty_matrix for NaN-tolerant scoring"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(arr, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)  # constant columns → 0, matching the loop's guard
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=X.columns, columns=X.columns)
