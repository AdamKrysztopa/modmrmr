"""Determinism and parity of the parallel pairwise driver."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.profiling.data import make_matrix
from modmrmr.core.scorers import as_penalty_matrix, get_scorer
from tests.parity import assert_parity

# mixed_ksg has no score_matrix fast path, so it actually exercises the
# parallel pair loop rather than short-circuiting to a BLAS call.
_LOOP_SCORER = "mixed_ksg"


def test_parallel_matches_serial_exactly():
    scorer = get_scorer(_LOOP_SCORER)
    X, _ = make_matrix(300, 10, "continuous", seed=71)
    serial = as_penalty_matrix(scorer, random_state=42, n_jobs=None)(X)
    parallel = as_penalty_matrix(scorer, random_state=42, n_jobs=-1)(X)
    assert_parity(
        serial.to_numpy(),
        parallel.to_numpy(),
        rtol=0.0,
        atol=0.0,
        systematic_atol=0.0,
        label="parallel-vs-serial",
    )


def test_parallel_is_stable_across_repeated_runs():
    scorer = get_scorer(_LOOP_SCORER)
    X, _ = make_matrix(300, 10, "continuous", seed=72)
    first = as_penalty_matrix(scorer, random_state=42, n_jobs=-1)(X).to_numpy()
    second = as_penalty_matrix(scorer, random_state=42, n_jobs=-1)(X).to_numpy()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("n_jobs", [None, 1, 2, -1])
def test_all_n_jobs_settings_agree(n_jobs):
    scorer = get_scorer(_LOOP_SCORER)
    X, _ = make_matrix(200, 8, "continuous", seed=73)
    reference = as_penalty_matrix(scorer, random_state=42, n_jobs=None)(X).to_numpy()
    actual = as_penalty_matrix(scorer, random_state=42, n_jobs=n_jobs)(X).to_numpy()
    np.testing.assert_array_equal(reference, actual)


def test_labels_are_preserved_under_parallelism():
    X, _ = make_matrix(200, 6, "continuous", seed=74)
    out = as_penalty_matrix(get_scorer(_LOOP_SCORER), random_state=42, n_jobs=-1)(X)
    assert list(out.index) == list(X.columns)
    assert list(out.columns) == list(X.columns)


def test_default_is_serial():
    """A library must not grab every core without being asked."""
    import inspect

    sig = inspect.signature(as_penalty_matrix)
    assert sig.parameters["n_jobs"].default is None
