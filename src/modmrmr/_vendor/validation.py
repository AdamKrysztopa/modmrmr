"""Validation helpers for time-series inputs and run configuration."""

from __future__ import annotations

import numpy as np


def validate_time_series(
    ts: np.ndarray,
    *,
    min_length: int,
) -> np.ndarray:
    """Validate a univariate time series.

    Args:
        ts: Input series.
        min_length: Minimum valid length.

    Returns:
        np.ndarray: Flattened validated series.

    Raises:
        ValueError: If the series is invalid.
    """
    if min_length < 2:
        raise ValueError("min_length must be >= 2")

    arr = np.asarray(ts, dtype=float).reshape(-1)

    if arr.size < min_length:
        raise ValueError("Time series is too short.")

    if not np.isfinite(arr).all():
        raise ValueError("Time series contains NaN or inf.")

    if np.std(arr) == 0:
        raise ValueError("Time series is constant.")

    return arr
