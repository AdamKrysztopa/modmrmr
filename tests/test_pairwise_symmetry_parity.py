"""Parity of the symmetry-exploiting pairwise drivers against the full loop."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.profiling.data import make_matrix
from modmrmr.core.scorers import as_penalty_matrix, get_scorer
from tests.parity import assert_parity


# Reference: the full ordered-pair double loop, as it stood before Task 8.
def _reference_penalty(scorer, X, random_state: int):
    cols = list(X.columns)
    arrs = {c: X[c].to_numpy(dtype=float) for c in cols}
    import pandas as pd

    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            out.loc[a, b] = (
                1.0
                if i == j
                else scorer.score_pair(arrs[a], arrs[b], random_state=random_state).raw_value
            )
    return out


_SYMMETRIC = ["pearson_abs", "spearman_abs", "gcmi"]


@pytest.mark.parametrize("name", _SYMMETRIC)
@pytest.mark.parametrize("kind", ["continuous", "discrete"])
def test_symmetric_driver_matches_full_loop(name: str, kind: str):
    scorer = get_scorer(name)
    X, _ = make_matrix(400, 10, kind, seed=13)
    reference = _reference_penalty(scorer, X, 42)
    fast = as_penalty_matrix(scorer, random_state=42)(X)

    assert list(fast.index) == list(reference.index)
    assert list(fast.columns) == list(reference.columns)
    assert_parity(
        reference.to_numpy(),
        fast.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        systematic_atol=1e-13,
        label=f"penalty:{name}:{kind}",
    )


@pytest.mark.parametrize("name", _SYMMETRIC)
def test_output_matrix_is_actually_symmetric(name: str):
    X, _ = make_matrix(300, 8, "continuous", seed=14)
    out = as_penalty_matrix(get_scorer(name), random_state=42)(X).to_numpy()
    np.testing.assert_allclose(out, out.T, rtol=0, atol=0)


def test_asymmetric_scorer_still_computes_both_directions():
    """tree_r2 is not symmetric; the driver must not mirror it.

    Seed chosen (not the brief's seed=15) because with this repo's locked
    scikit-learn==1.9.0, seed=15 at this shape makes every pairwise OOB R^2
    negative, which _TreeR2Scorer clips to 0.0 in both directions -- a
    degenerate, trivially-symmetric matrix unrelated to whether the driver
    mirrors. Verified this also happens on the pre-Task-8 baseline driver.
    """
    X, _ = make_matrix(200, 5, "continuous", seed=4)
    scorer = get_scorer("tree_r2")
    out = as_penalty_matrix(scorer, random_state=42)(X).to_numpy()
    off = out[~np.eye(5, dtype=bool)]
    assert not np.allclose(out, out.T), (
        "tree_r2 output is symmetric, which means the driver mirrored a "
        "scorer that declares symmetric=False."
    )
    assert np.isfinite(off).all()


def test_symmetric_driver_halves_the_call_count():
    """The whole point: p(p-1)/2 calls instead of p(p-1)."""
    calls = {"n": 0}
    base = get_scorer("pearson_abs")

    class _Counting:
        name = "counting"
        symmetric = True
        supports_redundancy = True

        def score_pair(self, x, y, *, random_state=42):
            calls["n"] += 1
            return base.score_pair(x, y, random_state=random_state)

    X, _ = make_matrix(100, 10, "continuous", seed=16)
    as_penalty_matrix(_Counting(), random_state=42)(X)
    assert calls["n"] == 10 * 9 // 2
