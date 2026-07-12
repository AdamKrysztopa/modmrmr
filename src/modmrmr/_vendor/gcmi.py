"""Gaussian Copula Mutual Information (GCMI) diagnostics.

Implements the Ince et al. (2017) bivariate GCMI estimator.  GCMI is
monotonic-transform invariant: rank-copula normalization Gaussianizes the
marginals before the closed-form Gaussian MI is applied.  Unlike kNN MI,
GCMI has no random state — it is fully deterministic.

Reference:
    Ince, R. A. A., et al. (2017). A statistical framework for
    neuroimaging data analysis based on mutual information estimated with
    Gaussian copula. *Human Brain Mapping*, 38(3), 1541–1573.
    https://doi.org/10.1002/hbm.23471
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

from modmrmr._vendor.validation import validate_time_series

# Clip bound for Pearson rho to prevent log(0) in the MI formula.
_RHO_CLIP = 1.0 - 1e-9


def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """Rank-normalize *x* to Gaussian via the probit transform.

    Implements the mapping :math:`u_i = \\operatorname{rank}(x_i) / (n+1)`,
    then :math:`z_i = \\Phi^{-1}(u_i)`.  The ``(n+1)`` denominator ensures
    ``u_i ∈ (0, 1)`` so the probit is always finite.

    Args:
        x: 1-D float array.

    Returns:
        Gaussianized array of the same shape as *x*.
    """
    n = x.size
    u = rankdata(x) / (n + 1)
    return norm.ppf(u)


def _gcmi_bivariate(x_g: np.ndarray, y_g: np.ndarray) -> float:
    """Compute MI in bits from two already-Gaussianized arrays.

    Uses the bivariate Gaussian formula:

    .. math::

        I(X; Y) = -\\frac{1}{2} \\log_2(1 - \\rho^2)

    where :math:`\\rho` is the Pearson correlation of the Gaussianized arrays.
    The result is always non-negative since :math:`|\\rho| \\le 1`.

    Args:
        x_g: Gaussianized version of the first array.
        y_g: Gaussianized version of the second array, aligned to *x_g*.

    Returns:
        Non-negative MI estimate in bits.
    """
    rho = float(np.corrcoef(x_g, y_g)[0, 1])
    if not np.isfinite(rho):
        return 0.0
    rho_clipped = np.clip(abs(rho), 0.0, _RHO_CLIP)
    mi_bits = -0.5 * np.log2(1.0 - rho_clipped**2)
    return max(float(mi_bits), 0.0)


def compute_gcmi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_pairs: int = 30,
) -> float:
    """Compute Gaussian Copula MI (GCMI) between two arrays.

    GCMI is monotonic-transform invariant: it uses rank-copula normalization
    to Gaussianize marginals before computing the closed-form Gaussian MI.
    Unlike kNN MI, GCMI has no random state — it is fully deterministic.

    Args:
        x: First array.
        y: Second array, aligned to *x*.
        min_pairs: Minimum sample pairs required.

    Returns:
        Non-negative GCMI estimate in bits.

    Raises:
        ValueError: If arrays are too short, misaligned, or invalid.
    """
    x_arr = validate_time_series(x, min_length=min_pairs)
    y_arr = validate_time_series(y, min_length=min_pairs)
    if x_arr.size != y_arr.size:
        raise ValueError(f"x and y must have identical lengths; got {x_arr.size} and {y_arr.size}")
    return _gcmi_bivariate(_rank_normalise(x_arr), _rank_normalise(y_arr))


def compute_gcmi_at_lag(
    source: np.ndarray,
    target: np.ndarray,
    *,
    lag: int,
    min_pairs: int = 30,
) -> float:
    """Compute GCMI between ``source_{t-lag}`` and ``target_t``.

    The zero-lag variant is symmetric: ``compute_gcmi(x, y) == compute_gcmi(y, x)``.
    Cross-lag MI is directional in pairing — swapping source and target with the
    same positive lag changes which observations are aligned and is not guaranteed
    to be equal.

    Args:
        source: Source series.
        target: Target series, same length as *source*.
        lag: The lag to apply to *source* before computing MI.
        min_pairs: Minimum aligned sample pairs.

    Returns:
        Non-negative GCMI estimate in bits.

    Raises:
        ValueError: If *lag* < 1 or arrays are invalid / too short.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1; got {lag}")
    min_length = lag + min_pairs
    src = validate_time_series(source, min_length=min_length)
    tgt = validate_time_series(target, min_length=min_length)
    if src.size != tgt.size:
        raise ValueError(
            f"source and target must have identical lengths; got {src.size} and {tgt.size}"
        )
    src_lagged = src[: src.size - lag]
    tgt_aligned = tgt[lag:]
    return _gcmi_bivariate(_rank_normalise(src_lagged), _rank_normalise(tgt_aligned))


def _compute_gcmi_curve_validated(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    max_lag: int,
    min_pairs: int,
) -> np.ndarray:
    """Run the per-lag GCMI loop on already-validated equal-length arrays.

    Per-lag rank-normalization is preserved inside the loop because each lag
    yields a different aligned paired sample; rank-copula must reflect that
    lag-specific sample.
    """
    result = np.empty(max_lag, dtype=float)
    for h in range(1, max_lag + 1):
        src_lagged = src[: src.size - h]
        tgt_aligned = tgt[h:]
        result[h - 1] = _gcmi_bivariate(_rank_normalise(src_lagged), _rank_normalise(tgt_aligned))
    return result


def compute_gcmi_curve(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_lag: int,
    min_pairs: int = 30,
) -> np.ndarray:
    """Compute GCMI for lags 1 .. *max_lag*.

    Args:
        source: Source series.
        target: Target series, same length as *source*.
        max_lag: Maximum lag to evaluate.
        min_pairs: Minimum aligned sample pairs.

    Returns:
        1-D array of shape ``(max_lag,)`` with GCMI per lag (index 0 = lag 1).

    Raises:
        ValueError: If *max_lag* < 1 or arrays are invalid.
    """
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1; got {max_lag}")
    min_length = max_lag + min_pairs
    src = validate_time_series(source, min_length=min_length)
    tgt = validate_time_series(target, min_length=min_length)
    if src.size != tgt.size:
        raise ValueError(
            f"source and target must have identical lengths; got {src.size} and {tgt.size}"
        )
    return _compute_gcmi_curve_validated(src, tgt, max_lag=max_lag, min_pairs=min_pairs)
