"""Tests for the profiling harness synthetic data generators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.profiling.data import make_matrix, make_pair


def test_make_pair_shapes_and_determinism():
    x1, y1 = make_pair(200, "continuous", seed=7)
    x2, y2 = make_pair(200, "continuous", seed=7)
    assert x1.shape == (200,)
    assert y1.shape == (200,)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_give_different_data():
    x1, _ = make_pair(200, "continuous", seed=7)
    x2, _ = make_pair(200, "continuous", seed=8)
    assert not np.array_equal(x1, x2)


def test_discrete_kind_has_heavy_ties():
    """The discrete generator must produce many exact duplicates.

    This is the whole point of the kind: it is what makes the tie branch in
    _mixed_ksg_mi execute. A generator that produced distinct floats would
    make the O(n^2) hot spot invisible to the profiler.
    """
    x, y = make_pair(500, "discrete", seed=1)
    assert len(np.unique(x)) <= 20
    assert len(np.unique(y)) <= 20


def test_mixed_kind_is_continuous_x_discrete_y():
    x, y = make_pair(500, "mixed", seed=1)
    assert len(np.unique(x)) > 400
    assert len(np.unique(y)) <= 20


def test_all_kinds_are_finite():
    for kind in ("continuous", "discrete", "mixed"):
        x, y = make_pair(300, kind, seed=3)
        assert np.isfinite(x).all()
        assert np.isfinite(y).all()


def test_pair_is_dependent_not_independent():
    """A profiling workload of independent noise would be unrepresentative."""
    x, y = make_pair(2000, "continuous", seed=5)
    assert abs(np.corrcoef(x, y)[0, 1]) > 0.2


def test_make_matrix_shapes_and_determinism():
    X1, y1 = make_matrix(150, 12, "continuous", seed=4)
    X2, y2 = make_matrix(150, 12, "continuous", seed=4)
    assert isinstance(X1, pd.DataFrame)
    assert X1.shape == (150, 12)
    assert y1.shape == (150,)
    pd.testing.assert_frame_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_make_matrix_columns_are_correlated():
    """Redundancy between features must exist or the penalty path is trivial."""
    X, _ = make_matrix(1000, 8, "continuous", seed=2)
    corr = X.corr().abs().to_numpy()
    off_diagonal = corr[~np.eye(8, dtype=bool)]
    assert off_diagonal.max() > 0.3
