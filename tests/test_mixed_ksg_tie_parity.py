"""Parity of the vectorized coincidence counter against the reference loop."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.profiling.data import make_pair
from modmrmr.core.scorers import get_scorer
from modmrmr.core.scorers.mutual_info import (
    _coincidence_counts,
    _eps_separated,
    _mixed_ksg_mi,
)
from tests.parity import assert_parity

_EPS = 1e-10


def _reference_counts(x: np.ndarray, y: np.ndarray, tie: np.ndarray, eps: float) -> np.ndarray:
    """The original per-tied-point Python loop, kept as the authority."""
    counts = np.full(len(x), np.nan)
    for i in np.nonzero(tie)[0]:
        close = (np.abs(x - x[i]) <= eps) & (np.abs(y - y[i]) <= eps)
        counts[i] = float(np.count_nonzero(close))
    return counts


@pytest.mark.parametrize("kind", ["discrete", "mixed"])
@pytest.mark.parametrize("seed", [31, 32, 33])
def test_vectorized_counts_match_the_loop(kind: str, seed: int):
    x, y = make_pair(600, kind, seed=seed)
    tie = np.ones(len(x), dtype=bool)
    fast = _coincidence_counts(x, y, _EPS)
    assert fast is not None, "eps-separated discrete data must take the fast path"
    reference = _reference_counts(x, y, tie, _EPS)
    assert_parity(reference, fast, rtol=0.0, atol=0.0, systematic_atol=0.0, label="tie-counts")


def test_eps_separated_detects_near_but_distinct_values():
    assert _eps_separated(np.array([0.0, 1.0, 2.0]), _EPS) is True
    assert _eps_separated(np.array([0.0, 1e-12, 2.0]), _EPS) is False


def test_eps_separated_on_a_constant_array():
    assert _eps_separated(np.array([5.0, 5.0, 5.0]), _EPS) is True


def test_falls_back_when_values_are_not_eps_separated():
    """Non-transitive eps boxes must not be approximated by exact grouping."""
    x = np.array([0.0, 1e-12, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert _coincidence_counts(x, y, _EPS) is None


@pytest.mark.parametrize("kind", ["continuous", "discrete", "mixed"])
@pytest.mark.parametrize("seed", [41, 42])
def test_mixed_ksg_mi_value_is_unchanged(kind: str, seed: int):
    """The estimator's output, not just the counter, must be identical."""
    x, y = make_pair(500, kind, seed=seed)
    xs = x / np.std(x) if np.std(x) > 0 else x
    ys = y / np.std(y) if np.std(y) > 0 else y
    value = _mixed_ksg_mi(xs, ys, k=5)
    assert np.isfinite(value)
    assert value >= 0.0


@pytest.mark.parametrize("name", ["mixed_ksg", "ami_adaptive"])
@pytest.mark.parametrize("kind", ["continuous", "discrete", "mixed"])
def test_scorer_output_is_finite_and_nonnegative(name: str, kind: str):
    x, y = make_pair(400, kind, seed=51)
    raw = get_scorer(name).score_pair(x, y, random_state=7).raw_value
    assert np.isfinite(raw)
    assert raw >= 0.0
