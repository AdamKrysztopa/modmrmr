"""Linear/monotonic pairwise dependence scorers."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore


class _PearsonAbsScorer:
    """Absolute Pearson correlation.  Cheap linear baseline."""

    name: str = "pearson_abs"

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


class _SpearmanAbsScorer:
    """Absolute Spearman rank correlation.  Monotonic nonparametric baseline."""

    name: str = "spearman_abs"

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
