"""Linear/monotonic pairwise dependence scorers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore, _symmetrize


def _finite_matrix(X: pd.DataFrame) -> np.ndarray:
    """Return ``X`` as a float array, rejecting non-finite input.

    The matrix fast paths transform whole columns at once, so a single NaN
    would contaminate every pair involving that column — whereas the pairwise
    reference masks per pair. Rejecting here makes the driver fall back to the
    NaN-tolerant loop rather than silently returning different numbers.
    """
    arr = X.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(
            "score_matrix requires fully finite input; the pairwise "
            "score_pair path is the NaN-tolerant reference."
        )
    return arr


def _abs_corr_frame(arr: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """|Pearson| matrix of ``arr``'s columns, with the scorers' zero conventions."""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(arr, rowvar=False))
    # A constant column yields NaN correlation; score_pair returns 0.0 there.
    corr = np.nan_to_num(corr, nan=0.0)
    corr = _symmetrize(corr)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=X.columns, columns=X.columns)


class _PearsonAbsScorer:
    """Absolute Pearson correlation.  Cheap linear baseline."""

    name: str = "pearson_abs"
    # |rho| is invariant under argument order; scipy computes it symmetrically.
    symmetric: bool = True

    def score_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,
    ) -> _RawScore:
        n = int(np.sum(np.isfinite(x) & np.isfinite(y)))
        warn: list[str] = []
        if n < _MIN_PAIRS:
            warn.append(f"Only {n} finite pairs; score set to 0.")
            return _RawScore(raw_value=0.0, n_pairs=n, warnings=warn)
        mask = np.isfinite(x) & np.isfinite(y)
        xf, yf = x[mask], y[mask]
        result = pearsonr(xf, yf)
        raw = float(abs(result.statistic))
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=raw,
            n_pairs=n,
            estimator_settings={"scorer": "pearson_abs"},
        )

    def score_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """All-pairs |Pearson| in one BLAS call.

        Exact fast path for the p^2 score_pair loop: np.corrcoef computes the
        same statistic scipy.stats.pearsonr does, for every pair at once.
        """
        return _abs_corr_frame(_finite_matrix(X), X)


class _SpearmanAbsScorer:
    """Absolute Spearman rank correlation.  Monotonic nonparametric baseline."""

    name: str = "spearman_abs"
    # Ranks are computed per-argument, then correlated symmetrically.
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
        result = spearmanr(xf, yf)
        raw = float(abs(result.statistic))
        if not math.isfinite(raw):
            raw = 0.0
        return _RawScore(
            raw_value=raw,
            n_pairs=n,
            estimator_settings={"scorer": "spearman_abs"},
        )

    def score_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """All-pairs |Spearman| in one BLAS call.

        Spearman is Pearson on ranks, so ranking each column once and taking
        one correlation matrix reproduces the pairwise result exactly.
        """
        arr = _finite_matrix(X)
        ranks = np.apply_along_axis(rankdata, 0, arr)
        return _abs_corr_frame(ranks, X)
