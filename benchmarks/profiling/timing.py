"""Timing primitives for the profiling sweep.

Reports median and inter-quartile range rather than mean and standard
deviation. Wall-clock timings on a laptop are contaminated by GC pauses and
thermal throttling, both of which are one-sided: they can only make a run
slower. A mean absorbs those outliers into the reported figure, whereas a
median rejects them and the IQR makes their presence visible.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimingResult:
    """Outcome of one timed measurement.

    Attributes:
        median_s: Median wall-clock seconds across repeats.
        iqr_s: Inter-quartile range in seconds; a large value relative to the
            median means the measurement is contaminated and untrustworthy.
        repeats: Number of timed (non-warmup) runs.
        total_s: Total wall-clock seconds spent, warmup included. Used by the
            sweep driver for ETA estimation.
    """

    median_s: float
    iqr_s: float
    repeats: int
    total_s: float


def measure(
    fn: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: int = 1,
    _perf_counter: Callable[[], float] = time.perf_counter,
) -> TimingResult:
    """Time ``fn`` over ``repeats`` runs after ``warmup`` untimed runs.

    Args:
        fn: Zero-argument callable to time. Any return value is discarded.
        repeats: Number of timed runs; must be >= 1.
        warmup: Number of untimed runs first, to pay one-off import, JIT, and
            cache-warming costs outside the measurement.
        _perf_counter: Clock injection point for tests. Not for production use.

    Returns:
        A :class:`TimingResult`.

    Raises:
        ValueError: If ``repeats`` < 1 or ``warmup`` < 0.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    start_total = _perf_counter()
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeats):
        start = _perf_counter()
        fn()
        samples.append(_perf_counter() - start)
    total = _perf_counter() - start_total

    arr = np.asarray(samples, dtype=float)
    q75, q25 = np.percentile(arr, [75, 25])
    return TimingResult(
        median_s=float(np.median(arr)),
        iqr_s=float(q75 - q25),
        repeats=repeats,
        total_s=float(total),
    )
