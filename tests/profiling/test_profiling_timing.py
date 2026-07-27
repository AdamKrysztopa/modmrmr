"""Tests for the profiling timing primitives."""

from __future__ import annotations

import pytest

from benchmarks.profiling.timing import TimingResult, measure


def test_measure_returns_timing_result():
    result = measure(lambda: sum(range(1000)), repeats=3, warmup=1)
    assert isinstance(result, TimingResult)
    assert result.repeats == 3
    assert result.median_s >= 0.0
    assert result.iqr_s >= 0.0
    assert result.total_s >= result.median_s


def test_measure_calls_fn_warmup_plus_repeats_times():
    calls = []
    measure(lambda: calls.append(1), repeats=4, warmup=2)
    assert len(calls) == 6


def test_measure_median_is_robust_to_a_single_outlier():
    """One slow run must not dominate: that is why this reports a median."""
    durations = [0.001, 0.001, 0.001, 0.001, 0.5]
    it = iter(durations)
    clock = {"t": 0.0}

    def fake_perf_counter():
        return clock["t"]

    def fn():
        clock["t"] += next(it)

    result = measure(fn, repeats=5, warmup=0, _perf_counter=fake_perf_counter)
    assert result.median_s == pytest.approx(0.001)


def test_measure_rejects_nonpositive_repeats():
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        measure(lambda: None, repeats=0)


def test_timing_result_is_frozen():
    result = measure(lambda: None, repeats=1, warmup=0)
    with pytest.raises(AttributeError):
        result.median_s = 1.0  # type: ignore[misc]
