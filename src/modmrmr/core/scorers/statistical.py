"""Statistical (F-test and classification-MI) pairwise relevance scorers.

These measures are relevance-oriented (feature-vs-target). Their raw values are
unbounded (F-statistics in ``[0, inf)``, MI in nats); the ``tmrmr`` pool path
rescales them via its ``rank_percentile`` normalizer, but ``MRMRSelector`` uses
the raw values directly (unbounded relevance is fine there — the greedy loop only
compares relevances by ``idxmax``).

``f_classif``/``mutual_info_classif`` require *discrete class labels* as the
second argument, so they are relevance-only (``supports_redundancy = False``):
feeding them feature-vs-feature through the redundancy adapter is meaningless and
is rejected by :func:`~modmrmr.core.scorers.base.as_penalty_matrix`.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore


class _FRegressionScorer:
    """Univariate linear F-test relevance (sklearn ``f_regression``)."""

    name: str = "f_regression"

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
        f_stat, p_val = f_regression(x[mask].reshape(-1, 1), y[mask])
        raw = float(f_stat[0])
        p = float(p_val[0])
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            p_value=p if math.isfinite(p) else None,
            estimator_settings={"scorer": "f_regression"},
        )


class _FClassifScorer:
    """ANOVA F-test relevance for classification targets (sklearn ``f_classif``)."""

    name: str = "f_classif"
    supports_redundancy: bool = False

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
        labels = y[mask]
        if np.unique(labels).size < 2:
            return _RawScore(
                raw_value=0.0, n_pairs=n, warnings=["Fewer than 2 classes; score set to 0."]
            )
        f_stat, p_val = f_classif(x[mask].reshape(-1, 1), labels)
        raw = float(f_stat[0])
        p = float(p_val[0])
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            p_value=p if math.isfinite(p) else None,
            estimator_settings={"scorer": "f_classif"},
        )


class _MutualInfoClassifScorer:
    """Classification mutual information relevance (sklearn ``mutual_info_classif``).

    ``y`` is integer-encoded before scoring because the pairwise adapter feeds
    targets through as floats; MI-for-classification needs discrete class labels.
    """

    name: str = "mutual_info_classif"
    supports_redundancy: bool = False

    def __init__(self, *, n_neighbors: int = 3) -> None:
        self._n_neighbors = n_neighbors

    def with_neighbors(self, n_neighbors: int) -> _MutualInfoClassifScorer:
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
        raw = float(
            mutual_info_classif(
                x[mask].reshape(-1, 1),
                labels,
                n_neighbors=self._n_neighbors,
                random_state=random_state,
            )[0]
        )
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            estimator_settings={
                "scorer": "mutual_info_classif",
                "n_neighbors": self._n_neighbors,
            },
        )
