"""B-spline soft-binning mutual information scorer.

Daub, Steuer, Selbig & Kloska (2004), "Estimating mutual information using
B-spline functions -- an improved similarity measure for analysing gene
expression data". Each variable is min-max scaled to ``[0, 1]`` and replaced
by its fractional ("soft") membership across ``n_bins`` overlapping B-spline
basis functions -- a partition of unity, so every sample's per-variable
weights sum to 1 exactly like a one-hot histogram bin, except fractional.
The soft joint/marginal histograms are then plugged into the standard
discrete mutual-information formula. This is a deterministic plug-in
estimator: no resampling, no RNG.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import BSpline

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore

_EPS = 1e-12


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """Scale ``values`` to ``[0, 1]``; a constant column maps to a constant 0.5.

    The 0.5 fallback keeps every sample in a single interior bin (well inside
    the spline's support) rather than dividing by zero, so a degenerate
    (zero-range) input still yields a finite, well-defined basis matrix.
    """
    lo = float(values.min())
    hi = float(values.max())
    rng = hi - lo
    if rng <= _EPS:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / rng


def _bspline_basis_matrix(scaled: np.ndarray, *, n_bins: int, order: int) -> np.ndarray:
    """Soft-bin membership matrix for values already scaled to ``[0, 1]``.

    Returns an ``(n, n_bins)`` matrix of non-negative B-spline basis weights.
    Uses a clamped/open-uniform knot vector over ``[0, 1]`` with ``n_bins``
    basis functions of the given spline ``order`` (piecewise-polynomial
    degree ``order - 1``), so the basis functions form a partition of unity
    (rows sum to 1) by construction, including at the ``0.0``/``1.0``
    boundaries. A residual-slop renormalization guards floating-point error.
    """
    degree = order - 1
    n_internal = n_bins - order
    if n_internal < 0:
        raise ValueError(f"n_bins ({n_bins}) must be >= order ({order})")
    internal_knots = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    knots = np.concatenate([np.zeros(order), internal_knots, np.ones(order)])
    n_coef = len(knots) - degree - 1  # == n_bins

    basis = np.zeros((scaled.shape[0], n_bins), dtype=float)
    for i in range(n_coef):
        coef = np.zeros(n_coef)
        coef[i] = 1.0
        spline = BSpline(knots, coef, degree, extrapolate=False)
        basis[:, i] = np.nan_to_num(spline(scaled), nan=0.0)

    row_sums = basis.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > _EPS, row_sums, 1.0)
    return basis / row_sums


def _mi_from_joint(joint: np.ndarray, p_row: np.ndarray, p_col: np.ndarray) -> float:
    """Plug-in discrete MI from a soft joint histogram and its marginals."""
    outer = np.outer(p_row, p_col)
    mask = joint > _EPS
    safe_outer = np.where(outer > _EPS, outer, 1.0)
    ratio = np.where(mask, joint / safe_outer, 1.0)
    terms = np.where(mask, joint * np.log(ratio), 0.0)
    return float(terms.sum())


def _continuous_mi(x_scaled: np.ndarray, y_scaled: np.ndarray, *, n_bins: int, order: int) -> float:
    """MI between two continuous variables, both soft-binned via B-splines."""
    w_x = _bspline_basis_matrix(x_scaled, n_bins=n_bins, order=order)
    w_y = _bspline_basis_matrix(y_scaled, n_bins=n_bins, order=order)
    n = x_scaled.shape[0]
    joint = (w_x.T @ w_y) / n  # (n_bins, n_bins)
    p_i = w_x.mean(axis=0)
    p_j = w_y.mean(axis=0)
    return _mi_from_joint(joint, p_i, p_j)


def _discrete_mi(x_scaled: np.ndarray, y: np.ndarray, *, n_bins: int, order: int) -> float:
    """MI between a soft-binned continuous ``x`` and a discrete ``y``.

    ``y``'s classes are treated as exact one-hot bins (no spline on ``y``),
    which is the group-by-class form of the estimator: joint p(i, c) is the
    mean, over samples in class ``c``, of x's soft membership in bin ``i``,
    weighted by that class's frequency.
    """
    w_x = _bspline_basis_matrix(x_scaled, n_bins=n_bins, order=order)
    n = x_scaled.shape[0]
    classes, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
    joint = np.zeros((n_bins, classes.shape[0]), dtype=float)
    for c in range(classes.shape[0]):
        joint[:, c] = w_x[y_idx == c, :].sum(axis=0) / n
    p_i = w_x.mean(axis=0)
    p_c = counts / n
    return _mi_from_joint(joint, p_i, p_c)


class _BSplineMiScorer:
    """B-spline soft-binning mutual information (Daub, Steuer, Selbig & Kloska 2004).

    Deterministic plug-in MI estimator with no resampling: min-max scale each
    variable to ``[0, 1]``, replace each sample by its fractional ("soft")
    membership across ``n_bins`` overlapping B-spline basis functions (a
    partition of unity), and plug the resulting soft joint/marginal
    histograms into the discrete MI formula.

    ``degree`` follows the classical B-spline **order** convention (order k
    is a piecewise polynomial of degree k - 1); the default ``degree=3``
    therefore builds an order-3 (quadratic, ``scipy`` polynomial degree 2)
    basis, matching Daub et al.'s reference configuration.

    For a discrete target (few unique values relative to sample size), the
    y-side spline is skipped and exact one-hot class indicators are used
    instead (the group-by-class form) -- this avoids the confound of
    treating a discrete target as continuous.

    Coarse binning (``n_bins=8``) is known to underestimate MI at high
    dependence (the soft bins blur fine joint structure); this bias grows
    with the true MI and is a property of the estimator, not a bug.
    """

    name: str = "bspline_mi"

    def __init__(self, *, n_bins: int = 8, degree: int = 3) -> None:
        self._n_bins = n_bins
        self._degree = degree

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

        x_scaled = _minmax_scale(np.asarray(xf, dtype=float))
        if y_is_discrete:
            raw = _discrete_mi(x_scaled, yf, n_bins=self._n_bins, order=self._degree)
        else:
            y_scaled = _minmax_scale(np.asarray(yf, dtype=float))
            raw = _continuous_mi(x_scaled, y_scaled, n_bins=self._n_bins, order=self._degree)

        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            estimator_settings={
                "scorer": "bspline_mi",
                "n_bins": self._n_bins,
                "degree": self._degree,
            },
        )
