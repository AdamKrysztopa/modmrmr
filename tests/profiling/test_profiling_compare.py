"""Tests for the before/after comparison and gate evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.profiling.compare import evaluate_gate, fit_exponent, speedup_table


def _scaling_frame(rows):
    return pd.DataFrame(rows, columns=["scorer", "data_kind", "n", "median_s", "status"])


def test_fit_exponent_recovers_a_quadratic_kernel():
    df = _scaling_frame(
        [("slow", "discrete", 100, 1.0, "ok"), ("slow", "discrete", 200, 4.0, "ok")]
    )
    out = fit_exponent(df)
    assert out.loc[0, "exponent"] == pytest.approx(2.0, rel=0.02)


def test_fit_exponent_recovers_a_linear_kernel():
    df = _scaling_frame(
        [("fast", "continuous", 100, 1.0, "ok"), ("fast", "continuous", 200, 2.0, "ok")]
    )
    assert fit_exponent(df).loc[0, "exponent"] == pytest.approx(1.0, rel=0.02)


def test_fit_exponent_ignores_skipped_rows():
    df = _scaling_frame(
        [
            ("s", "continuous", 100, 1.0, "ok"),
            ("s", "continuous", 200, 2.0, "ok"),
            ("s", "continuous", 400, None, "skipped_projected_cost"),
        ]
    )
    out = fit_exponent(df)
    assert out.loc[0, "n_points"] == 2


def test_speedup_table_reports_the_ratio():
    before = _scaling_frame([("s", "continuous", 100, 4.0, "ok")])
    after = _scaling_frame([("s", "continuous", 100, 1.0, "ok")])
    out = speedup_table(before, after)
    assert out.loc[0, "speedup"] == pytest.approx(4.0)


def test_gate_fails_when_no_kernel_dominates():
    """Amdahl: a 30% kernel cannot justify a toolchain."""
    end_to_end = pd.DataFrame(
        {"scorer": ["a", "b", "c"], "median_s": [3.0, 3.5, 3.5], "status": ["ok"] * 3}
    )
    exponents = pd.DataFrame({"scorer": ["a", "b", "c"], "exponent": [2.0, 2.0, 2.0]})
    verdicts = evaluate_gate(end_to_end, exponents)
    assert not verdicts["passes_gate"].any()


def test_gate_fails_for_third_party_bound_scorers():
    """tree_r2 is 200 sklearn tree fits; Rust cannot touch it.

    It is dominant and superlinear here, so only the third-party exclusion
    can fail it — which is exactly what this asserts.
    """
    end_to_end = pd.DataFrame(
        {"scorer": ["tree_r2", "x"], "median_s": [99.0, 1.0], "status": ["ok", "ok"]}
    )
    exponents = pd.DataFrame({"scorer": ["tree_r2", "x"], "exponent": [2.5, 1.0]})
    verdicts = evaluate_gate(end_to_end, exponents).set_index("scorer")
    assert bool(verdicts.loc["tree_r2", "dominates"]) is True
    assert bool(verdicts.loc["tree_r2", "superlinear"]) is True
    assert bool(verdicts.loc["tree_r2", "own_code"]) is False
    assert bool(verdicts.loc["tree_r2", "passes_gate"]) is False


def test_gate_passes_for_a_dominant_superlinear_own_kernel():
    end_to_end = pd.DataFrame(
        {"scorer": ["mixed_ksg", "x"], "median_s": [90.0, 10.0], "status": ["ok", "ok"]}
    )
    exponents = pd.DataFrame({"scorer": ["mixed_ksg", "x"], "exponent": [1.9, 1.0]})
    verdicts = evaluate_gate(end_to_end, exponents)
    assert bool(verdicts.set_index("scorer").loc["mixed_ksg", "passes_gate"])
