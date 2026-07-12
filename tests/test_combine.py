"""Unit tests for the pure combiner + aggregator (Plan A, Task 1)."""

import numpy as np
import pandas as pd
import pytest

from modmrmr.core.combine import aggregate, combine


def _block() -> pd.DataFrame:
    # rows = candidate (not-yet-selected) features, cols = already-selected features
    return pd.DataFrame(
        {"s1": [0.2, 0.8, 0.4], "s2": [0.6, 0.2, 0.4]},
        index=["c1", "c2", "c3"],
    )


def test_aggregate_mean() -> None:
    out = aggregate(_block(), "mean")
    pd.testing.assert_series_equal(out, pd.Series([0.4, 0.5, 0.4], index=["c1", "c2", "c3"]))


def test_aggregate_max() -> None:
    out = aggregate(_block(), "max")
    pd.testing.assert_series_equal(out, pd.Series([0.6, 0.8, 0.4], index=["c1", "c2", "c3"]))


def test_aggregate_sum() -> None:
    out = aggregate(_block(), "sum")
    pd.testing.assert_series_equal(out, pd.Series([0.8, 1.0, 0.8], index=["c1", "c2", "c3"]))


def test_aggregate_skips_nan() -> None:
    block = pd.DataFrame({"s1": [np.nan, 0.8], "s2": [0.6, np.nan]}, index=["c1", "c2"])
    pd.testing.assert_series_equal(
        aggregate(block, "mean"), pd.Series([0.6, 0.8], index=["c1", "c2"])
    )


def test_aggregate_rejects_unknown_how() -> None:
    import pytest

    with pytest.raises(ValueError, match="aggregation"):
        aggregate(_block(), "median")


def test_combine_difference() -> None:
    rel = pd.Series([1.0, 0.5], index=["c1", "c2"])
    agg = pd.Series([0.3, 0.1], index=["c1", "c2"])
    pd.testing.assert_series_equal(
        combine(rel, agg, "difference"), pd.Series([0.7, 0.4], index=["c1", "c2"])
    )


def test_combine_quotient_applies_floor() -> None:
    rel = pd.Series([1.0, 2.0], index=["c1", "c2"])
    # c2's aggregated redundancy (0.0) is floored to 1e-3 before dividing.
    agg = pd.Series([0.5, 0.0], index=["c1", "c2"])
    out = combine(rel, agg, "quotient")
    assert out.loc["c1"] == 2.0
    assert out.loc["c2"] == 2000.0


def test_combine_reg_quotient_adds_epsilon() -> None:
    rel = pd.Series([1.0, 2.0], index=["c1", "c2"])
    # score = rel / (red + 0.1): c1 -> 1/(0.5+0.1)=1.6667, c2 -> 2/(0.0+0.1)=20.
    agg = pd.Series([0.5, 0.0], index=["c1", "c2"])
    out = combine(rel, agg, "reg_quotient")
    assert out.loc["c1"] == pytest.approx(1.0 / 0.6)
    assert out.loc["c2"] == pytest.approx(20.0)


def test_combine_reg_quotient_fills_nan_with_zero() -> None:
    # NaN redundancy -> 0, so score = rel / eps (never divides by ~0 like the raw
    # quotient's floor); this bounded denominator is the point of the arm.
    rel = pd.Series([3.0], index=["c1"])
    out = combine(rel, pd.Series([np.nan], index=["c1"]), "reg_quotient")
    assert out.loc["c1"] == pytest.approx(30.0)


def test_combine_multiplicative_clips_at_zero() -> None:
    rel = pd.Series([1.0, 1.0], index=["c1", "c2"])
    # c2's redundancy exceeds 1, so (1 - r) would be negative -> clipped to 0.
    agg = pd.Series([0.25, 1.5], index=["c1", "c2"])
    out = combine(rel, agg, "multiplicative")
    assert out.loc["c1"] == 0.75
    assert out.loc["c2"] == 0.0


def test_combine_quotient_fills_nan_with_floor() -> None:
    rel = pd.Series([3.0], index=["c1"])
    out = combine(rel, pd.Series([np.nan], index=["c1"]), "quotient")
    assert out.loc["c1"] == 3000.0


def test_combine_multiplicative_fills_nan_with_zero() -> None:
    rel = pd.Series([3.0], index=["c1"])
    out = combine(rel, pd.Series([np.nan], index=["c1"]), "multiplicative")
    assert out.loc["c1"] == 3.0  # (1 - 0) = 1


def test_combine_difference_fills_nan_with_zero() -> None:
    rel = pd.Series([3.0], index=["c1"])
    out = combine(rel, pd.Series([np.nan], index=["c1"]), "difference")
    assert out.loc["c1"] == 3.0


def test_combine_rejects_unknown_operator() -> None:
    import pytest

    rel = pd.Series([1.0], index=["c1"])
    agg = pd.Series([0.1], index=["c1"])
    with pytest.raises(ValueError, match="operator"):
        combine(rel, agg, "ratio")
