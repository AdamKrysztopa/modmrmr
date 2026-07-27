"""Parity assertions for optimized fast paths.

Every fast path added in Phase 2 must reproduce the reference implementation.
"Reproduce" is checked two ways, because element-wise closeness alone is not
sufficient evidence:

1. **Element-wise** — catches gross errors and wrong formulas.
2. **Systematic drift** — the mean *signed* difference must be near zero.
   Vectorization changes float accumulation order, which produces symmetric
   jitter with no preferred sign. A consistent one-directional offset is
   instead evidence that the fast path computes something subtly different,
   and it can hide comfortably inside a loose element-wise tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityReport:
    """Measured agreement between a reference and a fast implementation."""

    max_abs_diff: float
    mean_signed_diff: float
    n_compared: int


def assert_parity(
    reference: np.ndarray,
    fast: np.ndarray,
    *,
    rtol: float,
    atol: float,
    systematic_atol: float,
    label: str,
) -> ParityReport:
    """Assert ``fast`` reproduces ``reference`` within tolerance.

    Args:
        reference: Values from the reference (slow, authoritative) path.
        fast: Values from the optimized path.
        rtol: Relative tolerance for the element-wise check.
        atol: Absolute tolerance for the element-wise check.
        systematic_atol: Bound on the mean signed difference. Set this much
            tighter than ``atol`` — it is testing for bias, not for noise.
        label: Name used in assertion messages.

    Returns:
        A :class:`ParityReport` describing the measured agreement.

    Raises:
        AssertionError: On shape mismatch, differing non-finite positions,
            element-wise divergence, or systematic drift.
    """
    ref = np.asarray(reference, dtype=float)
    fst = np.asarray(fast, dtype=float)

    assert ref.shape == fst.shape, f"[{label}] shape mismatch: {ref.shape} vs {fst.shape}"

    ref_finite = np.isfinite(ref)
    fst_finite = np.isfinite(fst)
    assert np.array_equal(ref_finite, fst_finite), (
        f"[{label}] non-finite values occupy different positions: "
        f"{int((ref_finite != fst_finite).sum())} disagreeing entries"
    )

    r = ref[ref_finite]
    f = fst[ref_finite]
    diff = f - r
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_signed = float(np.mean(diff)) if diff.size else 0.0

    assert np.allclose(r, f, rtol=rtol, atol=atol), (
        f"[{label}] element-wise parity failed: max|diff|={max_abs:.3e} "
        f"exceeds rtol={rtol:.3e}, atol={atol:.3e}"
    )
    assert abs(mean_signed) <= systematic_atol, (
        f"[{label}] systematic drift: mean signed diff={mean_signed:.3e} "
        f"exceeds systematic_atol={systematic_atol:.3e}. Float reassociation "
        f"is symmetric; a consistent offset means the fast path differs."
    )

    return ParityReport(max_abs_diff=max_abs, mean_signed_diff=mean_signed, n_compared=int(r.size))
