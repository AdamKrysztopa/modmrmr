"""ReliefF relevance scorer (skrebate).

RELEVANCE ONLY. ReliefF scores a feature by how its values separate the target
between near neighbours — a feature-vs-target quantity with no symmetric
feature-vs-feature meaning. ``supports_redundancy = False`` makes the redundancy
adapter (``as_penalty_matrix``) refuse it instead of returning a meaningless
matrix. Raw weights fall in ``[-1, 1]``; the ``tmrmr`` pool path rescales them to
``[0, 1]`` via its ``rank_percentile`` normalizer, while ``MRMRSelector`` compares
the raw weights directly by ``idxmax`` (a negative weight simply ranks last).
"""

from __future__ import annotations

import math

import numpy as np
from skrebate import ReliefF

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore


class _ReliefFScorer:
    """ReliefF feature relevance (skrebate). Relevance only — not a redundancy."""

    name: str = "relieff"
    supports_redundancy: bool = False

    def __init__(self, *, n_neighbors: int = 10) -> None:
        self._n_neighbors = n_neighbors

    def with_neighbors(self, n_neighbors: int) -> _ReliefFScorer:
        """Return a copy tuned to ``n_neighbors`` (frozen ``with_neighbors`` convention)."""
        return type(self)(n_neighbors=n_neighbors)

    def score_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,
    ) -> _RawScore:
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < _MIN_PAIRS:
            return _RawScore(
                raw_value=0.0, n_pairs=n, warnings=[f"Only {n} finite pairs; score set to 0."]
            )
        _, labels = np.unique(y[mask], return_inverse=True)
        k = int(min(self._n_neighbors, n - 1))
        relief = ReliefF(n_neighbors=max(k, 1), n_features_to_select=1)
        relief.fit(x[mask].reshape(-1, 1), labels)
        raw = float(relief.feature_importances_[0])
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=raw,
            n_pairs=n,
            estimator_settings={"scorer": "relieff", "n_neighbors": k},
        )
