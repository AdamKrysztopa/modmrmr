"""Parity and fallback behaviour of the matrix-level fast paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.profiling.data import make_matrix
from modmrmr.core.scorers import as_penalty_matrix, get_scorer
from tests.parity import assert_parity

# Per-scorer tolerances. pearson and spearman are one corrcoef either way, so
# they agree to near machine precision. gcmi applies -0.5*log2(1-rho^2), which
# amplifies error as rho -> 1, hence the looser bound.
_TOLERANCES = {
    "pearson_abs": dict(rtol=1e-11, atol=1e-12, systematic_atol=1e-13),
    "spearman_abs": dict(rtol=1e-11, atol=1e-12, systematic_atol=1e-13),
    "gcmi": dict(rtol=1e-9, atol=1e-10, systematic_atol=1e-11),
}


def _pairwise_reference(scorer, X: pd.DataFrame) -> np.ndarray:
    cols = list(X.columns)
    arrs = {c: X[c].to_numpy(dtype=float) for c in cols}
    out = np.empty((len(cols), len(cols)), dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            out[i, j] = (
                1.0 if i == j else scorer.score_pair(arrs[a], arrs[b], random_state=42).raw_value
            )
    return out


@pytest.mark.parametrize("name", sorted(_TOLERANCES))
@pytest.mark.parametrize("kind", ["continuous", "discrete"])
def test_score_matrix_matches_pairwise_reference(name: str, kind: str):
    scorer = get_scorer(name)
    X, _ = make_matrix(500, 12, kind, seed=21)
    reference = _pairwise_reference(scorer, X)
    # copy=True: pandas 3 hands back a read-only view, and we overwrite the diagonal.
    fast = scorer.score_matrix(X).to_numpy(copy=True)
    np.fill_diagonal(fast, 1.0)
    assert_parity(reference, fast, label=f"matrix:{name}:{kind}", **_TOLERANCES[name])


@pytest.mark.parametrize("name", sorted(_TOLERANCES))
def test_score_matrix_preserves_labels(name: str):
    X, _ = make_matrix(200, 6, "continuous", seed=22)
    out = get_scorer(name).score_matrix(X)
    assert list(out.index) == list(X.columns)
    assert list(out.columns) == list(X.columns)


@pytest.mark.parametrize("name", sorted(_TOLERANCES))
def test_score_matrix_rejects_non_finite_input(name: str):
    X, _ = make_matrix(200, 5, "continuous", seed=23)
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        get_scorer(name).score_matrix(X)


@pytest.mark.parametrize("name", sorted(_TOLERANCES))
def test_driver_uses_the_fast_path_when_available(name: str):
    """as_penalty_matrix must not call score_pair at all for these scorers."""
    inner = get_scorer(name)
    calls = {"n": 0}

    class _Wrapped:
        name = "wrapped"
        symmetric = True
        supports_redundancy = True

        def score_matrix(self, X):
            return inner.score_matrix(X)

        def score_pair(self, x, y, *, random_state=42):
            calls["n"] += 1
            return inner.score_pair(x, y, random_state=random_state)

    X, _ = make_matrix(200, 8, "continuous", seed=24)
    as_penalty_matrix(_Wrapped(), random_state=42)(X)
    assert calls["n"] == 0


def test_driver_falls_back_to_the_loop_on_non_finite_input():
    """NaN tolerance is the reference path's job; the fast path must yield to it."""
    X, _ = make_matrix(200, 6, "continuous", seed=25)
    X.iloc[3, 2] = np.nan
    out = as_penalty_matrix(get_scorer("pearson_abs"), random_state=42)(X)
    assert np.isfinite(out.to_numpy()).all()


def test_fast_path_agrees_with_existing_fast_pearson_penalty():
    """The new generalization must not contradict the precedent it generalizes."""
    from modmrmr.core.scorers import fast_pearson_penalty

    X, _ = make_matrix(400, 10, "continuous", seed=26)
    existing = fast_pearson_penalty(X).to_numpy()
    new = get_scorer("pearson_abs").score_matrix(X).to_numpy(copy=True)
    np.fill_diagonal(new, 1.0)
    assert_parity(
        existing, new, rtol=1e-12, atol=1e-13, systematic_atol=1e-14, label="vs-fast-pearson"
    )
