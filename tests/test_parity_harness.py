"""Tests for the parity assertion harness itself."""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity import ParityReport, assert_parity

_KW = {"rtol": 1e-9, "atol": 1e-12, "systematic_atol": 1e-12, "label": "test"}


def test_identical_arrays_pass():
    a = np.linspace(0.0, 1.0, 50)
    report = assert_parity(a, a.copy(), **_KW)
    assert isinstance(report, ParityReport)
    assert report.max_abs_diff == 0.0
    assert report.n_compared == 50


def test_float_noise_within_tolerance_passes():
    rng = np.random.default_rng(0)
    a = rng.random(200)
    # Symmetric jitter: emulates reassociation, which has no preferred sign.
    b = a + rng.normal(0.0, 1e-13, size=a.shape)
    assert_parity(a, b, rtol=1e-9, atol=1e-11, systematic_atol=1e-11, label="jitter")


def test_gross_error_fails():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.5])
    with pytest.raises(AssertionError, match="element-wise"):
        assert_parity(a, b, **_KW)


def test_systematic_drift_fails_even_when_elementwise_passes():
    """The failure mode this harness exists to catch.

    A uniform 1e-7 bias is invisible to a loose allclose but means the fast
    path is computing something different.
    """
    rng = np.random.default_rng(1)
    a = rng.random(500)
    b = a + 1e-7
    with pytest.raises(AssertionError, match="systematic"):
        assert_parity(a, b, rtol=1e-3, atol=1e-3, systematic_atol=1e-9, label="drift")


def test_shape_mismatch_fails():
    with pytest.raises(AssertionError, match="shape"):
        assert_parity(np.zeros(3), np.zeros(4), **_KW)


def test_nan_in_same_positions_is_allowed():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    assert_parity(a, b, **_KW)


def test_nan_in_different_positions_fails():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 2.0, np.nan])
    with pytest.raises(AssertionError, match="non-finite"):
        assert_parity(a, b, **_KW)
