"""The lazy-redundancy fast path must be exact, and must stay gated."""

from __future__ import annotations

import pytest

from modmrmr.tmrmr.selector import _redundancy_is_pool_independent


class _Spec:
    def __init__(self, normalization: str) -> None:
        self.normalization = normalization


class _Config:
    def __init__(self, normalization: str) -> None:
        self.redundancy_scorer = _Spec(normalization)


def test_pass_through_normalization_is_pool_independent():
    assert _redundancy_is_pool_independent(_Config("none")) is True


def test_rank_percentile_is_not_pool_independent():
    """The whole reason laziness is gated: percentiles depend on the pool."""
    assert _redundancy_is_pool_independent(_Config("rank_percentile")) is False


@pytest.mark.parametrize(
    "strategy",
    ["surrogate_effect_clip", "nmi_min_entropy", "nmi_mean_entropy"],
)
def test_unproven_strategies_default_to_not_pool_independent(strategy: str):
    """A new strategy must be proven pool-independent before being allowed."""
    assert _redundancy_is_pool_independent(_Config(strategy)) is False
