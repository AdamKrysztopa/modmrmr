"""Pure combiner + aggregator functions for the configurable greedy mRMR engine.

These functions are the deep-module boundaries (2) and (3) from the design spec:
the greedy selector delegates all per-step scoring here so that every mRMR grid
cell (operator x aggregation) is expressible without touching the loop.
"""

from __future__ import annotations

import pandas as pd

# Quotient denominator floor: prevents divide-by-(near)-zero when the aggregated
# redundancy is tiny. Matches the legacy MRMR `_MEAN_REDUNDANCY_FLOOR`.
_QUOTIENT_FLOOR = 1e-3

# Regularized-quotient additive epsilon: score = rel / (red + eps). Unlike the
# hard floor above (which only clips near-zero denominators), this additively
# regularizes *every* candidate, interpolating between the raw quotient (eps->0)
# and pure relevance ranking (eps large, denominator ~ constant). At eps = 0.1 on
# the normalized [0, 1] redundancy scale it is the standard D/(R+eps) form used to
# probe whether regularization rescues the quotient's high-dimensional collapse.
_REG_QUOTIENT_EPS = 0.1

# Canonical enumerations — the single source of truth for the valid operator and
# aggregation labels. The estimator imports these to validate its params early.
AGGREGATIONS = ("mean", "max", "sum")
OPERATORS = ("difference", "quotient", "multiplicative", "reg_quotient")


def aggregate(redundancy_block: pd.DataFrame, how: str) -> pd.Series:
    """Collapse the already-selected redundancy columns into one value per candidate.

    Args:
        redundancy_block: rows = candidate (not-yet-selected) features,
            columns = already-selected features; values = pairwise redundancy.
        how: one of ``"mean"``, ``"max"``, ``"sum"`` (over columns / ``axis=1``).

    Returns:
        A Series indexed by the candidate features. ``NaN`` cells are skipped
        (pandas ``skipna=True``); an all-``NaN`` row yields ``NaN`` for
        ``"mean"``/``"max"`` (pandas ``"sum"`` returns ``0.0``). Either way the
        value is filled downstream by :func:`combine` per operator — and ``0.0``
        coincides with every operator's fill target, so scores are unaffected.

    Raises:
        ValueError: if ``how`` is not a recognised aggregation.
    """
    if how == "mean":
        return redundancy_block.mean(axis=1)
    if how == "max":
        return redundancy_block.max(axis=1)
    if how == "sum":
        return redundancy_block.sum(axis=1)
    raise ValueError(f"Unknown aggregation {how!r}. Expected one of: {', '.join(AGGREGATIONS)}.")


def combine(relevance: pd.Series, aggregated_redundancy: pd.Series, operator: str) -> pd.Series:
    """Combine per-candidate relevance with aggregated redundancy into a score.

    Args:
        relevance: per-candidate relevance (higher = more informative about y).
        aggregated_redundancy: per-candidate redundancy vs the already-selected
            set (output of :func:`aggregate`). ``NaN`` is filled per operator.
        operator: one of ``"difference"``, ``"quotient"``, ``"reg_quotient"``,
            ``"multiplicative"``.

    Returns:
        A Series of scores; the greedy loop selects ``idxmax`` of this each step.

    Raises:
        ValueError: if ``operator`` is not recognised.
    """
    if operator == "difference":
        return relevance - aggregated_redundancy.fillna(0.0)
    if operator == "quotient":
        floored = aggregated_redundancy.fillna(_QUOTIENT_FLOOR).clip(lower=_QUOTIENT_FLOOR)
        return relevance / floored
    if operator == "reg_quotient":
        return relevance / (aggregated_redundancy.fillna(0.0) + _REG_QUOTIENT_EPS)
    if operator == "multiplicative":
        penalty = (1.0 - aggregated_redundancy.fillna(0.0)).clip(lower=0.0)
        return relevance * penalty
    raise ValueError(f"Unknown operator {operator!r}. Expected one of: {', '.join(OPERATORS)}.")
