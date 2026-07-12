"""Tree-based pairwise dependence scorer."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from modmrmr.core.scorers.base import _MIN_PAIRS, _RawScore


class _TreeR2Scorer:
    """Random-forest out-of-bag R^2 as a nonlinear dependence proxy (clipped to [0, 1]).

    Out-of-bag R^2 is a generalization estimate: it scores each sample using only
    the trees that did not train on it, so it does not saturate at 1.0 the way
    in-sample R^2 does for a high-capacity forest. Pure-noise pairs score ~0.
    """

    name: str = "tree_r2"

    def __init__(self, *, n_estimators: int = 200) -> None:
        self._n_estimators = n_estimators

    def score_pair(self, x: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> _RawScore:
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < _MIN_PAIRS:
            return _RawScore(
                raw_value=0.0, n_pairs=n, warnings=[f"Only {n} finite pairs; score set to 0."]
            )
        model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            bootstrap=True,
            oob_score=True,
            random_state=random_state,
        )
        model.fit(x[mask].reshape(-1, 1), y[mask])
        oob = float(getattr(model, "oob_score_", float("nan")))
        raw = min(max(oob, 0.0), 1.0) if np.isfinite(oob) else 0.0
        return _RawScore(
            raw_value=raw,
            n_pairs=n,
            estimator_settings={"scorer": "tree_r2", "n_estimators": self._n_estimators},
        )
