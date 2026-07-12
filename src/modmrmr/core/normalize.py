"""Pool-relative normalizers for raw scorer values.

This module supplies the normalization primitives used to map raw pairwise scores
onto a ``[0, 1]``-bounded range before they are consumed by the greedy
selection loop.
"""

from __future__ import annotations

import bisect
import math

from modmrmr.models import NormalizationStrategy

# ---------------------------------------------------------------------------
# Pool normalizers
# ---------------------------------------------------------------------------


class _PoolNormalizer:
    """Pool-relative normalizer for raw scorer values.

    Fits on a pool of raw scores collected from one selection run, then
    transforms individual values to a ``[0, 1]``-bounded range.  The fitted
    state is deterministic given the same pool.

    Supported strategies:

    - ``"rank_percentile"``: rank-percentile position in the pool.
    - ``"none"``: pass-through (raw value is used as-is; clip externally).

    Args:
        strategy: Normalization strategy to apply.
        fit_scope_id: Human-readable identifier for the normalization pool
            (e.g. ``"relevance_run_42"``).
    """

    def __init__(
        self,
        strategy: NormalizationStrategy,
        *,
        fit_scope_id: str = "",
    ) -> None:
        self._strategy = strategy
        self._fit_scope_id = fit_scope_id
        self._sorted_pool: list[float] = []
        self._fitted = False

    @property
    def strategy(self) -> NormalizationStrategy:
        """Normalization strategy."""
        return self._strategy

    @property
    def fit_scope_id(self) -> str:
        """Identifier for the pool used to fit this normalizer."""
        return self._fit_scope_id

    def fit(self, scores: list[float]) -> None:
        """Fit the normalizer on a pool of raw scores.

        Args:
            scores: Raw score values from one selection run pool.

        Raises:
            ValueError: If the pool is empty or contains non-finite values.
        """
        if len(scores) == 0:
            raise ValueError("Cannot fit normalizer on an empty pool")
        finite = [v for v in scores if math.isfinite(v)]
        if len(finite) == 0:
            raise ValueError("All pool scores are non-finite; cannot fit normalizer")
        self._sorted_pool = sorted(finite)
        self._fitted = True

    def transform(self, value: float) -> float:
        """Transform one raw value using the fitted pool.

        For ``"rank_percentile"`` the result is the fraction of pool values
        that are <= ``value`` (always in ``[0, 1]``).

        For ``"none"`` the raw value is returned unchanged; the caller is
        responsible for clipping to ``[0, 1]`` when used as a similarity
        penalty.

        Args:
            value: Raw scorer output to normalize.

        Returns:
            Normalized value.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._strategy == "none":
            return value if math.isfinite(value) else 0.0

        if not self._fitted:
            raise RuntimeError(
                "Normalizer must be fitted before calling transform(). Call fit(pool) first."
            )

        if self._strategy == "rank_percentile":
            if not math.isfinite(value):
                return 0.0
            rank = bisect.bisect_right(self._sorted_pool, value)
            return float(rank) / len(self._sorted_pool)

        if self._strategy in ("surrogate_effect_clip", "nmi_min_entropy", "nmi_mean_entropy"):
            raise NotImplementedError(
                f"Normalization strategy {self._strategy!r} requires additional "
                "scorer infrastructure not yet available in Phase 1. "
                "Use 'rank_percentile' or 'none' instead."
            )

        return value  # fallback — should not be reached for defined strategies


def build_normalizer(
    strategy: NormalizationStrategy,
    *,
    fit_scope_id: str = "",
) -> _PoolNormalizer:
    """Build a pool normalizer for the given strategy.

    Args:
        strategy: Normalization strategy to apply.
        fit_scope_id: Human-readable pool identifier for diagnostics.

    Returns:
        An unfitted :class:`_PoolNormalizer` instance.
    """
    return _PoolNormalizer(strategy, fit_scope_id=fit_scope_id)
