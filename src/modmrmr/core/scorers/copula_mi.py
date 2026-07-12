"""Empirical-copula entropy mutual-information scorer.

Rank-transforms each variable to its empirical copula (uniform margins via
``scipy.stats.rankdata``), then estimates the copula's negative entropy --
equivalently, the mutual information between the two variables, since MI is
copula-invariant -- by a plain 2-D histogram of the rank pairs on the unit
square:

    I(X; Y) = sum_{i,j} p(i, j) * log(p(i, j) / (p_u(i) * p_v(j)))

This is *distribution-free*: unlike ``gcmi`` (which fits a Gaussian copula
and reads off MI from its correlation), no parametric copula family is
assumed -- only that the histogram resolves the copula density well enough at
the chosen ``n_bins``. For a discrete target, the class indicators are used
directly as the v-partition (group-by-class) instead of ranking ``y``.

Deterministic plug-in estimator: rank transform and histogramming involve no
randomness, so ``random_state`` is accepted (for Protocol conformance) but
unused.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore

_EPS = 1e-12


def _copula_rank(values: np.ndarray) -> np.ndarray:
    """Empirical-copula (uniform-margin) rank transform: ``rankdata(v) / (n + 1)``."""
    n = values.shape[0]
    return rankdata(values, method="average") / (n + 1)


def _mi_from_joint(joint: np.ndarray, p_row: np.ndarray, p_col: np.ndarray) -> float:
    """Plug-in discrete MI from a joint histogram/probability table and its marginals."""
    outer = np.outer(p_row, p_col)
    mask = joint > _EPS
    safe_outer = np.where(outer > _EPS, outer, 1.0)
    ratio = np.where(mask, joint / safe_outer, 1.0)
    terms = np.where(mask, joint * np.log(ratio), 0.0)
    return float(terms.sum())


def _continuous_mi(x: np.ndarray, y: np.ndarray, *, n_bins: int) -> float:
    """MI between two continuous variables via a histogram of their empirical copula."""
    u = _copula_rank(x)
    v = _copula_rank(y)
    counts, _, _ = np.histogram2d(u, v, bins=n_bins, range=[[0.0, 1.0], [0.0, 1.0]])
    total = counts.sum()
    if total <= _EPS:
        return 0.0
    p = counts / total
    p_u = p.sum(axis=1)
    p_v = p.sum(axis=0)
    return _mi_from_joint(p, p_u, p_v)


def _discrete_mi(x: np.ndarray, y: np.ndarray, *, n_bins: int) -> float:
    """MI between a continuous ``x`` (rank-binned) and a discrete ``y`` (group-by-class)."""
    n = x.shape[0]
    u = _copula_rank(x)
    u_bin = np.minimum((u * n_bins).astype(int), n_bins - 1)
    classes, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
    joint = np.zeros((n_bins, classes.shape[0]), dtype=float)
    for c in range(classes.shape[0]):
        bin_counts = np.bincount(u_bin[y_idx == c], minlength=n_bins)
        joint[:, c] = bin_counts / n
    p_u = joint.sum(axis=1)
    p_c = counts / n
    return _mi_from_joint(joint, p_u, p_c)


class _CopulaMiScorer:
    """Empirical-copula entropy mutual information (distribution-free histogram plug-in).

    Rank-transforms each variable to its empirical copula (uniform margins),
    then estimates MI as the mutual information of that copula via a plain
    ``n_bins`` x ``n_bins`` 2-D histogram on the unit square. Unlike ``gcmi``,
    which assumes a Gaussian copula, this makes no parametric assumption
    about the dependence structure -- the only approximation is the
    histogram's bin resolution.

    For a discrete target (few unique values relative to sample size), the
    y-side is not ranked/binned; its exact class indicators are used as the
    v-partition instead (the group-by-class form).

    Deterministic: rank transform and histogramming involve no randomness.
    ``random_state`` is accepted for Protocol conformance but unused.
    """

    name: str = "copula_mi"

    def __init__(self, *, n_bins: int = 16) -> None:
        self._n_bins = n_bins

    def score_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,  # noqa: ARG002 -- deterministic plug-in; unused, kept for Protocol
    ) -> _RawScore:
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < _MIN_PAIRS:
            return _RawScore(
                raw_value=0.0, n_pairs=n, warnings=[f"Only {n} finite pairs; score set to 0."]
            )
        xf, yf = x[mask], y[mask]

        uniq = np.unique(yf)
        y_is_discrete = uniq.size <= max(2, min(20, n // 10))

        if y_is_discrete:
            raw = _discrete_mi(np.asarray(xf, dtype=float), yf, n_bins=self._n_bins)
        else:
            raw = _continuous_mi(
                np.asarray(xf, dtype=float), np.asarray(yf, dtype=float), n_bins=self._n_bins
            )

        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            estimator_settings={"scorer": "copula_mi", "n_bins": self._n_bins},
        )
