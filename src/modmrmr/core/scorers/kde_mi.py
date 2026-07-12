"""Gaussian-KDE plug-in mutual information scorer.

Standardizes each variable to zero mean / unit variance, fits Gaussian-kernel
density estimates (via ``scipy.stats.gaussian_kde``) to the joint and marginal
distributions, and plugs the resubstitution log-density ratio into the
mutual-information integral:

    I(X; Y) = E[ log p(x, y) - log p(x) - log p(y) ]

approximated by the sample mean over the (possibly subsampled) data -- a
resubstitution / plug-in KDE estimator. For a discrete target, MI is instead
estimated via the entropy decomposition ``H(X) - sum_c p(c) * H(X | Y=c)``,
each term itself a KDE differential-entropy resubstitution estimate
(``-mean log p(x_i)``) evaluated within class ``c``.

``gaussian_kde`` scales quadratically in sample count (covariance + pairwise
evaluation), so for ``n > 3000`` the pair is subsampled up front, seeded by
``random_state``, before either estimation path runs. This makes the
estimator's value formally depend on ``random_state`` above that threshold --
by design, for speed -- while remaining deterministic given a fixed seed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import gaussian_kde

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore

_EPS = 1e-12
_MAX_N = 3000


def _standardize(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance standardize; a constant column maps to all-zeros.

    The all-zeros fallback keeps a degenerate (zero-variance) input finite
    rather than dividing by zero; downstream KDE fits on an all-zero array
    have zero variance and are handled by the singular-covariance guards.
    """
    mean = float(values.mean())
    std = float(values.std())
    if std <= _EPS:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def _kde_entropy(values: np.ndarray, *, bw_method: str | float) -> float | None:
    """KDE differential-entropy resubstitution estimate, or ``None`` if degenerate.

    Returns ``None`` (rather than raising) when there are too few points,
    the sample has zero variance, ``gaussian_kde`` hits a singular
    covariance, or the resulting log-density is non-finite everywhere --
    callers treat ``None`` as "no usable estimate".
    """
    if values.shape[0] < 2 or float(np.std(values)) <= _EPS:
        return None
    try:
        kde = gaussian_kde(values, bw_method=bw_method)
        log_p = kde.logpdf(values)
    except (np.linalg.LinAlgError, ValueError):
        return None
    log_p = log_p[np.isfinite(log_p)]
    if log_p.size == 0:
        return None
    return float(-np.mean(log_p))


def _continuous_mi(x: np.ndarray, y: np.ndarray, *, bw_method: str | float) -> float:
    """Resubstitution MI plug-in estimate between two standardized continuous variables."""
    try:
        joint = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
        marg_x = gaussian_kde(x, bw_method=bw_method)
        marg_y = gaussian_kde(y, bw_method=bw_method)
        log_joint = joint.logpdf(np.vstack([x, y]))
        log_x = marg_x.logpdf(x)
        log_y = marg_y.logpdf(y)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    terms = log_joint - log_x - log_y
    terms = terms[np.isfinite(terms)]
    if terms.size == 0:
        return 0.0
    return float(np.mean(terms))


def _discrete_mi(x: np.ndarray, y: np.ndarray, *, bw_method: str | float) -> float:
    """MI between a standardized continuous ``x`` and a discrete ``y``.

    Uses the entropy decomposition ``H(x) - sum_c p(c) * H(x | y=c)``, with
    each (conditional) entropy a KDE resubstitution estimate. Classes with
    fewer than 2 points (or zero-variance x within the class) fall back to
    ``H(x | y=c) = H(x)`` -- they contribute no information rather than
    raising, since a KDE conditional-entropy estimate is undefined there.
    """
    n = x.shape[0]
    h_x = _kde_entropy(x, bw_method=bw_method)
    if h_x is None:
        return 0.0
    classes, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
    weighted_conditional = 0.0
    for c in range(classes.shape[0]):
        h_xc = _kde_entropy(x[y_idx == c], bw_method=bw_method)
        if h_xc is None:
            h_xc = h_x
        weighted_conditional += (counts[c] / n) * h_xc
    return h_x - weighted_conditional


class _KdeMiScorer:
    """Gaussian-KDE plug-in mutual information (resubstitution estimator).

    Fits ``scipy.stats.gaussian_kde`` to the standardized joint and marginal
    distributions and plugs the log-density ratio, evaluated at the sample
    points, into the MI integral. For a discrete target, MI is instead
    estimated via the entropy decomposition ``H(x) - sum_c p(c) * H(x | y=c)``,
    each conditional entropy a KDE resubstitution estimate over that class's
    x-values.

    ``gaussian_kde`` scales quadratically in sample count; for ``n > 3000``
    the pair is subsampled (seeded by ``random_state``) before fitting, so
    the estimator is deterministic given a fixed seed but formally depends
    on it above the subsample threshold.
    """

    name: str = "kde_mi"

    def __init__(self, *, bw_method: str | float = "scott") -> None:
        self._bw_method = bw_method

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
        xf, yf = x[mask], y[mask]

        uniq = np.unique(yf)
        y_is_discrete = uniq.size <= max(2, min(20, n // 10))

        if n > _MAX_N:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=_MAX_N, replace=False)
            xf, yf = xf[idx], yf[idx]

        x_std = _standardize(np.asarray(xf, dtype=float))
        if y_is_discrete:
            raw = _discrete_mi(x_std, yf, bw_method=self._bw_method)
        else:
            y_std = _standardize(np.asarray(yf, dtype=float))
            raw = _continuous_mi(x_std, y_std, bw_method=self._bw_method)

        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            estimator_settings={"scorer": "kde_mi", "bw_method": str(self._bw_method)},
        )
