"""Mutual-information-based pairwise dependence scorers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.stats import norm, rankdata
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors

from modmrmr._vendor.gcmi import _RHO_CLIP, compute_gcmi
from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore, _symmetrize


class _MutualInfoSklearnScorer:
    """Mutual information via sklearn KSG estimator.

    Fast practical baseline using ``n_neighbors=3`` (lighter than AMI's 8).
    """

    name: str = "mutual_info_sklearn"

    def __init__(self, *, n_neighbors: int = 3) -> None:
        self._n_neighbors = n_neighbors

    def with_neighbors(self, n_neighbors: int) -> _MutualInfoSklearnScorer:
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
        warn: list[str] = []
        if n < _MIN_PAIRS:
            warn.append(f"Only {n} finite pairs; score set to 0.")
            return _RawScore(raw_value=0.0, n_pairs=n, warnings=warn)
        xf, yf = x[mask], y[mask]
        raw = float(
            mutual_info_regression(
                xf.reshape(-1, 1),
                yf,
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
                "scorer": "mutual_info_sklearn",
                "n_neighbors": self._n_neighbors,
            },
        )


class _CattKnnMiScorer:
    """Catt-style AMI / kNN MI scorer.

    Uses the KSG mutual information estimator with ``n_neighbors=8``,
    aligned with the existing AMI computation lineage.

    This is the scientific native scoring mode for Lag-Aware ModMRMR.
    It provides nonlinear dependence estimates suitable for relevance
    scoring, selected-lag redundancy, and target-history novelty penalties.
    """

    name: str = "catt_knn_mi"

    def __init__(self, *, n_neighbors: int = 8) -> None:
        self._n_neighbors = n_neighbors

    def with_neighbors(self, n_neighbors: int) -> _CattKnnMiScorer:
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
        warn: list[str] = []
        if n < _MIN_PAIRS:
            warn.append(f"Only {n} finite pairs; score set to 0.")
            return _RawScore(raw_value=0.0, n_pairs=n, warnings=warn)
        xf, yf = x[mask], y[mask]
        raw = float(
            mutual_info_regression(
                xf.reshape(-1, 1),
                yf,
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
                "scorer": "catt_knn_mi",
                "n_neighbors": self._n_neighbors,
            },
        )


class _GcmiScorer:
    """Gaussian-copula MI scorer.  Nonlinear proxy; experimental."""

    name: str = "gcmi"
    # Each marginal is rank-normalised independently, then correlated.
    symmetric: bool = True

    def score_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,
    ) -> _RawScore:
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        warn: list[str] = []
        if n < _MIN_PAIRS:
            warn.append(f"Only {n} finite pairs; score set to 0.")
            return _RawScore(raw_value=0.0, n_pairs=n, warnings=warn)
        xf, yf = x[mask], y[mask]
        raw = compute_gcmi(xf, yf)
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            estimator_settings={"scorer": "gcmi"},
        )

    def score_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """All-pairs GCMI in one BLAS call.

        GCMI rank-normalises each marginal independently and then applies
        I = -0.5*log2(1 - rho^2) to their Pearson correlation. Normalising each
        column once and taking a single correlation matrix therefore reproduces
        the pairwise result exactly — the per-column transform never depends on
        the partner column.
        """
        arr = X.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(
                "score_matrix requires fully finite input; the pairwise "
                "score_pair path is the NaN-tolerant reference."
            )
        n = arr.shape[0]
        gaussianized = norm.ppf(np.apply_along_axis(rankdata, 0, arr) / (n + 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.corrcoef(gaussianized, rowvar=False)
        rho = _symmetrize(np.nan_to_num(rho, nan=0.0))
        clipped = np.clip(np.abs(rho), 0.0, _RHO_CLIP)
        mi = np.maximum(-0.5 * np.log2(1.0 - clipped**2), 0.0)
        # Self-MI is the clip bound (~14.4 bits), but the pairwise driver writes
        # 1.0 on the diagonal; match it, or the normalization pool would change.
        np.fill_diagonal(mi, 1.0)
        return pd.DataFrame(mi, index=X.columns, columns=X.columns)


def _counts_within_radius(
    sorted_vals: np.ndarray, vals: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Count, per point, how many of ``vals`` fall within ``radii`` (inclusive), self included.

    ``sorted_vals`` must be ``vals`` sorted ascending. Uses binary search on the
    sorted marginal so the per-point neighbour count is O(log n) instead of an
    O(n) scan, without needing a separate kNN structure for the 1-D marginal.
    """
    lo = np.searchsorted(sorted_vals, vals - radii, side="left")
    hi = np.searchsorted(sorted_vals, vals + radii, side="right")
    return (hi - lo).astype(np.float64)


def _eps_separated(v: np.ndarray, eps: float) -> bool:
    """True when no two *distinct* values of ``v`` lie within ``eps``.

    Under this condition ``|v - v[i]| <= eps`` is equivalent to ``v == v[i]``,
    which is what licenses replacing the per-point box count with exact-
    duplicate grouping.
    """
    unique = np.unique(v)
    return bool(unique.size < 2 or np.min(np.diff(unique)) > eps)


def _coincidence_counts(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray | None:
    """Per-point count of samples inside the ``eps`` box, vectorized.

    Replaces an O(n^2) Python loop with one O(n log n) ``np.unique`` pass.

    The loop's ``<= eps`` box is not a transitive relation in general, so exact
    grouping is only equivalent when both coordinates are eps-separated. When
    they are not, this returns ``None`` and the caller uses the reference loop
    — parity by construction rather than by approximation.

    Args:
        x: First standardized marginal.
        y: Second standardized marginal.
        eps: Coincidence tolerance.

    Returns:
        Float array of per-point counts, or ``None`` if the fast path does not
        apply.
    """
    if not (_eps_separated(x, eps) and _eps_separated(y, eps)):
        return None
    pairs = np.column_stack([x, y])
    _, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    # numpy 2.0 returned a column-shaped inverse for axis=0; reshape defensively
    # so this works across the supported numpy range.
    return counts[inverse.reshape(-1)].astype(np.float64)


def _mixed_ksg_mi(x: np.ndarray, y: np.ndarray, *, k: int) -> float:
    """Mixed-KSG mutual information (nats), Gao, Kannan, Oh & Viswanath (2017).

    For each sample, the distance to its k-th nearest neighbour in the joint
    (Chebyshev/max-norm) space sets a local radius. Marginal neighbour counts
    within that radius replace the plain KSG marginal counts, and exact
    coincidences (ties from a discrete variable) are detected per-sample and
    given a local neighbour count instead of the fixed ``k`` — so a discrete
    ``y`` is handled natively with no separate group-by-class code path.
    """
    n = len(x)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return 0.0
    joint = np.column_stack([x, y])

    nn_joint = NearestNeighbors(n_neighbors=k_eff + 1, metric="chebyshev").fit(joint)
    joint_dist, _ = nn_joint.kneighbors(joint)
    rho = joint_dist[:, k_eff]  # distance to the k-th neighbour, excluding self

    eps = 1e-10
    tie = rho == 0.0
    # Strict "<" is approximated as "<= (rho - eps)"; ties use "<= eps" so exact
    # coincidences (including self) are counted.
    radius = np.where(tie, eps, np.maximum(rho - eps, 0.0))

    order_x = np.argsort(x, kind="stable")
    order_y = np.argsort(y, kind="stable")
    nx = _counts_within_radius(x[order_x], x, radius)
    ny = _counts_within_radius(y[order_y], y, radius)

    k_i = np.full(n, float(k_eff))
    if np.any(tie):
        fast_counts = _coincidence_counts(x, y, eps)
        if fast_counts is not None:
            k_i[tie] = fast_counts[tie]
        else:
            # Reference path: the eps box is not an equivalence relation on this
            # data, so each tied point is counted individually.
            for i in np.nonzero(tie)[0]:
                close = (np.abs(x - x[i]) <= eps) & (np.abs(y - y[i]) <= eps)
                k_i[i] = float(np.count_nonzero(close))

    mi = float(digamma(n) + np.mean(digamma(k_i) - digamma(nx) - digamma(ny)))
    return max(mi, 0.0)


class _MixedKsgScorer:
    """Mixed KSG mutual information scorer (Gao, Kannan, Oh & Viswanath 2017).

    Natively handles discrete-continuous mixtures (e.g. a binary/discrete
    ``y``) via a local, per-sample neighbour count instead of a separate
    group-by-class estimator path. Default ``n_neighbors=5``.
    """

    name: str = "mixed_ksg"
    # Both marginals are standardized before the joint Chebyshev search, so
    # the estimator is order-independent.
    symmetric: bool = True

    def __init__(self, *, n_neighbors: int = 5) -> None:
        self._n_neighbors = n_neighbors

    def with_neighbors(self, n_neighbors: int) -> _MixedKsgScorer:
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
        warn: list[str] = []
        if n < _MIN_PAIRS:
            warn.append(f"Only {n} finite pairs; score set to 0.")
            return _RawScore(raw_value=0.0, n_pairs=n, warnings=warn)
        xf = np.asarray(x[mask], dtype=float)
        yf = np.asarray(y[mask], dtype=float)
        # Standardize BOTH marginals to unit variance so the joint Chebyshev
        # radius is scale-invariant and argument-order-independent. Dividing
        # each marginal by its own std preserves a discrete variable's exact
        # ties (a constant scaling maps equal values to equal values), so the
        # native discrete-y handling is unaffected. A zero-std (constant)
        # marginal is left unscaled to avoid a divide-by-zero.
        x_std = float(np.std(xf))
        y_std = float(np.std(yf))
        xs = xf / x_std if x_std > 0 else xf
        ys = yf / y_std if y_std > 0 else yf
        raw = _mixed_ksg_mi(xs, ys, k=self._n_neighbors)
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            estimator_settings={"scorer": "mixed_ksg", "n_neighbors": self._n_neighbors},
        )


class _AmiAdaptiveScorer:
    """Adaptive-neighbour-count Mixed-KSG mutual information scorer.

    Shares the audited ``_mixed_ksg_mi`` joint-kNN digamma core with
    ``mixed_ksg`` (Gao, Kannan, Oh & Viswanath 2017); the only difference is
    that ``k`` is chosen per-dataset from the sample size instead of fixed,
    via the standard density-adaptive rule ``k = clip(round(sqrt(n) / 2),
    k_min, k_max)``. Larger ``n`` -> more neighbours -> lower estimator
    variance, which avoids the noisy small-``k`` collapse a fixed low
    ``n_neighbors`` (e.g. 3) suffers on confounded regimes.
    """

    name: str = "ami_adaptive"
    # Both marginals are standardized before the joint Chebyshev search, so
    # the estimator is order-independent.
    symmetric: bool = True

    def __init__(self, *, k_min: int = 3, k_max: int = 20) -> None:
        self._k_min = k_min
        self._k_max = k_max

    def score_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,
    ) -> _RawScore:
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        warn: list[str] = []
        if n < _MIN_PAIRS:
            warn.append(f"Only {n} finite pairs; score set to 0.")
            return _RawScore(raw_value=0.0, n_pairs=n, warnings=warn)
        k = int(np.clip(round(np.sqrt(n) / 2), self._k_min, self._k_max))
        k = min(k, n - 1)  # kNN needs at least k other points
        xf = np.asarray(x[mask], dtype=float)
        yf = np.asarray(y[mask], dtype=float)
        # Standardize BOTH marginals to unit variance, matching _MixedKsgScorer,
        # so the joint Chebyshev radius is scale-invariant and
        # argument-order-independent; a constant marginal is left unscaled to
        # avoid a divide-by-zero.
        x_std = float(np.std(xf))
        y_std = float(np.std(yf))
        xs = xf / x_std if x_std > 0 else xf
        ys = yf / y_std if y_std > 0 else yf
        raw = _mixed_ksg_mi(xs, ys, k=k)
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=max(raw, 0.0),
            n_pairs=n,
            estimator_settings={
                "scorer": "ami_adaptive",
                "k": k,
                "k_min": self._k_min,
                "k_max": self._k_max,
            },
        )
