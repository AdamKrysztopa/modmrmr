import numpy as np
import pytest

from modmrmr._vendor.gcmi import compute_gcmi
from modmrmr._vendor.validation import validate_time_series


def test_validate_rejects_constant_series() -> None:
    with pytest.raises(ValueError):
        validate_time_series(np.ones(50), min_length=10)


def test_validate_returns_flattened_float_array() -> None:
    out = validate_time_series(np.arange(50).reshape(-1, 1), min_length=10)
    assert out.shape == (50,)
    assert out.dtype == float


def test_gcmi_detects_linear_dependence() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 2.0 * x + rng.normal(scale=0.1, size=400)
    independent = rng.normal(size=400)
    assert compute_gcmi(x, y) > compute_gcmi(x, independent)
