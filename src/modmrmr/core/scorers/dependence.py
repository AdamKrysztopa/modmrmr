"""General nonlinear dependence pairwise scorers (distance correlation, RDC).

Both measures are bounded in ``[0, 1]`` and marginal-invariant, so they may be
used as relevance OR as pairwise redundancy without further normalization.
"""

from __future__ import annotations

import math

import dcor
import numpy as np
from scipy.stats import rankdata

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore

# Ridge for regularized CCA inside RDC — stabilizes the rank-deficient copula
# feature covariances against spurious unit canonical correlations.
_RDC_RIDGE = 1e-6


class _DistanceCorrScorer:
    """Distance correlation (Szekely 2007) via the ``dcor`` package.

    Zero iff the variables are independent; one for a perfect (possibly
    nonlinear monotone) functional relationship. Always in ``[0, 1]``.
    """

    name: str = "distance_corr"

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
        raw = float(dcor.distance_correlation(x[mask], y[mask]))
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=min(max(raw, 0.0), 1.0),
            n_pairs=n,
            estimator_settings={"scorer": "distance_corr"},
        )


def _rdc_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 20,
    s: float = 1.0 / 6.0,
    random_state: int = 42,
) -> float:
    """Randomized Dependence Coefficient (Lopez-Paz 2013).

    rank-transform -> empirical copula -> random Gaussian projections ->
    sin/cos nonlinearity -> largest canonical correlation via numpy.
    """
    rng = np.random.default_rng(random_state)
    n = x.shape[0]
    # 1. Copula transform: empirical CDF via AVERAGE ranks, scaled to (0, 1].
    #    Average (not ordinal) ranks so tied/constant columns map to a constant
    #    copula (RDC ~ 0) instead of a spurious monotone ramp that fabricates
    #    dependence.
    cx = rankdata(x, method="average") / n
    cy = rankdata(y, method="average") / n
    # 2. Augment each copula with a bias/intercept column.
    x1 = np.column_stack([cx, np.ones(n)])
    y1 = np.column_stack([cy, np.ones(n)])
    # 3. Random Gaussian linear projections, scaled by s.
    px = x1 @ (rng.standard_normal((x1.shape[1], k)) * s)
    py = y1 @ (rng.standard_normal((y1.shape[1], k)) * s)
    # 4. Non-linear sin/cos random feature maps (2k features per side).
    zx = np.hstack([np.sin(px), np.cos(px)])
    zy = np.hstack([np.sin(py), np.cos(py)])
    m = zx.shape[1]
    # 5. Canonical correlation between the two feature blocks (numpy eig).
    cov = np.cov(np.hstack([zx, zy]), rowvar=False)
    cxx, cyy = cov[:m, :m], cov[m:, m:]
    cxy, cyx = cov[:m, m:], cov[m:, :m]
    # The sin/cos features are functions of a rank-2 copula projection, so the
    # within-block covariances (cxx, cyy) are heavily rank-deficient; a bare pinv
    # amplifies near-null directions and yields spurious canonical correlations
    # of ~1 even for independent (and monotone-vs-noise) inputs. A tiny ridge
    # (regularized CCA) stabilizes the inverse: independence stays ~0.1, genuine
    # dependence stays ~1.
    ridge = _RDC_RIDGE * np.eye(m)
    mat = np.linalg.pinv(cxx + ridge) @ cxy @ np.linalg.pinv(cyy + ridge) @ cyx
    eigs = np.linalg.eigvals(mat)
    real_eigs = np.real(eigs[np.abs(eigs.imag) < 1e-9])
    if real_eigs.size == 0:
        return 0.0
    # Squared canonical correlations lie in [0, 1]; numerical error can push the
    # largest slightly outside, so CLIP into range. Filtering (dropping eigs > 1)
    # would discard the strongest dependence exactly when it matters, collapsing
    # RDC to ~0 for near-perfect relationships.
    clipped = np.clip(real_eigs, 0.0, 1.0)
    return float(np.sqrt(np.max(clipped)))


class _RdcScorer:
    """Randomized Dependence Coefficient (Lopez-Paz 2013), bounded in ``[0, 1]``.

    Marginal-invariant nonlinear dependence: correlation of random non-linear
    copula projections approximating the HGR maximum correlation.
    """

    name: str = "rdc"

    def __init__(self, *, k: int = 20, s: float = 1.0 / 6.0) -> None:
        self._k = k
        self._s = s

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
        raw = _rdc_coefficient(x[mask], y[mask], k=self._k, s=self._s, random_state=random_state)
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=min(max(raw, 0.0), 1.0),
            n_pairs=n,
            estimator_settings={"scorer": "rdc", "k": self._k, "s": self._s},
        )
