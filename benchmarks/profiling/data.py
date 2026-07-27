"""Synthetic data generators for the profiling sweep.

Three data kinds exist because scorer cost depends on data character, not only
on size. In particular the tie branch of ``_mixed_ksg_mi`` only executes when
the joint k-th-neighbour distance is exactly zero, which requires exact
duplicates — so a purely continuous sweep would never profile it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

DataKind = Literal["continuous", "discrete", "mixed"]

# Number of distinct levels used by the discrete generators. Small enough that
# a sample of a few hundred produces heavy exact-duplicate ties.
_N_LEVELS = 8
# Correlation injected between x and y (and between feature columns) so the
# profiled workload has real dependence rather than independent noise.
_DEPENDENCE = 0.6


def _correlated_normal(
    rng: np.random.Generator, n: int, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return two standard-normal arrays with Pearson correlation ~= ``rho``."""
    base = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    partner = rho * base + np.sqrt(max(1.0 - rho**2, 0.0)) * noise
    return base, partner


def _discretize(values: np.ndarray, n_levels: int = _N_LEVELS) -> np.ndarray:
    """Map continuous values onto ``n_levels`` equal-count integer levels."""
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_levels + 1)[1:-1])
    return np.searchsorted(quantiles, values).astype(float)


def make_pair(n: int, kind: DataKind, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate one dependent ``(x, y)`` pair of length ``n``.

    Args:
        n: Number of samples.
        kind: ``"continuous"`` (both float), ``"discrete"`` (both level-coded),
            or ``"mixed"`` (continuous x, discrete y).
        seed: Seed for this cell; callers derive it from the cell index.

    Returns:
        Two finite 1-D float arrays of length ``n``.
    """
    rng = np.random.default_rng(seed)
    x, y = _correlated_normal(rng, n, _DEPENDENCE)
    if kind == "continuous":
        return x, y
    if kind == "discrete":
        return _discretize(x), _discretize(y)
    if kind == "mixed":
        return x, _discretize(y)
    raise ValueError(f"Unknown data kind {kind!r}. Expected continuous, discrete, or mixed.")


def make_matrix(n: int, p: int, kind: DataKind, *, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a feature matrix and target for driver / end-to-end benchmarks.

    Columns share a latent factor so the redundancy matrix is non-trivial; a
    matrix of independent columns would make every redundancy penalty ~0 and
    would not exercise the selection loop realistically.

    Args:
        n: Number of samples.
        p: Number of features.
        kind: Data character, as in :func:`make_pair`.
        seed: Seed for this cell.

    Returns:
        ``(X, y)`` with ``X`` a ``(n, p)`` DataFrame with columns ``f0..f{p-1}``.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    noise = rng.standard_normal((n, p))
    raw = _DEPENDENCE * latent[:, None] + np.sqrt(max(1.0 - _DEPENDENCE**2, 0.0)) * noise
    target = _DEPENDENCE * latent + np.sqrt(max(1.0 - _DEPENDENCE**2, 0.0)) * rng.standard_normal(n)

    if kind in ("discrete", "mixed"):
        target = _discretize(target)
    if kind == "discrete":
        raw = np.column_stack([_discretize(raw[:, j]) for j in range(p)])

    X = pd.DataFrame(raw, columns=[f"f{j}" for j in range(p)])
    return X, target
