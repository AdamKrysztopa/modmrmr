"""Registry-wide audit of the declared `symmetric` flag against measured behaviour.

A scorer's `symmetric` flag licenses the pairwise drivers to compute one
triangle and mirror it. If the flag is wrong, redundancy matrices silently
change. This test is the evidence for every flag in the registry.
"""

from __future__ import annotations

import pytest

from benchmarks.profiling.data import make_pair
from modmrmr.core.scorers import get_scorer, list_scorers

# Scorers that cannot act as a redundancy penalty are exempt: the pairwise
# drivers never call them in both directions.
_RELEVANCE_ONLY = {"relieff"}

# Measured asymmetry below this is float noise, not an implementation
# difference. Deliberately loose relative to parity tolerances: an
# implementation asymmetry (noise injection, regression direction) shows up
# orders of magnitude above this.
_SYMMETRY_ATOL = 1e-9


def _max_asymmetry(name: str) -> float:
    scorer = get_scorer(name)
    worst = 0.0
    for kind in ("continuous", "discrete", "mixed"):
        for seed in (11, 22, 33):
            x, y = make_pair(300, kind, seed=seed)
            forward = scorer.score_pair(x, y, random_state=7).raw_value
            reverse = scorer.score_pair(y, x, random_state=7).raw_value
            worst = max(worst, abs(forward - reverse))
    return worst


def _redundancy_scorers() -> list[str]:
    return [
        n
        for n in list_scorers()
        if n not in _RELEVANCE_ONLY and getattr(get_scorer(n), "supports_redundancy", True)
    ]


@pytest.mark.parametrize("name", _redundancy_scorers())
def test_declared_symmetry_matches_measured_behaviour(name: str):
    scorer = get_scorer(name)
    declared = getattr(scorer, "symmetric", False)
    measured = _max_asymmetry(name)

    if declared:
        assert measured <= _SYMMETRY_ATOL, (
            f"{name} declares symmetric=True but score_pair(x,y) and "
            f"score_pair(y,x) differ by up to {measured:.3e}. Either the flag "
            f"is wrong or the implementation is argument-order dependent."
        )
    else:
        assert measured > _SYMMETRY_ATOL, (
            f"{name} declares symmetric=False but is measurably symmetric "
            f"(max diff {measured:.3e}). Set symmetric=True to claim the 2x "
            f"saving in the pairwise drivers."
        )


def test_at_least_the_correlation_family_is_symmetric():
    """Guards against the audit trivially passing with every flag set False."""
    for name in ("pearson_abs", "spearman_abs", "gcmi"):
        assert getattr(get_scorer(name), "symmetric", False) is True
