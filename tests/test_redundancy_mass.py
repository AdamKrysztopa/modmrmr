import numpy as np

from analysis.redundancy_mass import _synthetic_independent_columns, _tail_slope, redundancy_mass


def _pareto_sample(rng: np.random.Generator, alpha: float, n: int) -> np.ndarray:
    """Classical Type-I Pareto(alpha) samples with x_m = 1 -- tail index is alpha exactly."""
    u = rng.uniform(size=n)
    return (1.0 - u) ** (-1.0 / alpha)


def _independent(n=300, p=200, seed=0):
    """Mutually independent columns: the W -> 0 regime the quotient is exposed to."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * rng.standard_normal(n)  # only column 0 is relevant
    return X, y


def _correlated(n=300, p=200, seed=0):
    """Dense low-rank columns (the isolet regime): W stays bounded away from zero."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 5))
    loadings = rng.standard_normal((5, p))
    X = base @ loadings + 0.1 * rng.standard_normal((n, p))
    y = base[:, 0] + 0.5 * rng.standard_normal(n)
    return X, y


def test_independent_columns_concentrate_redundancy_near_zero():
    X, y = _independent()
    rm = redundancy_mass(X, y, k_selected=10, random_state=0)
    assert rm.mass_below_abs > 0.5
    assert rm.w_median < 0.10


def test_correlated_columns_keep_redundancy_away_from_zero():
    X, y = _correlated()
    rm = redundancy_mass(X, y, k_selected=10, random_state=0)
    assert rm.mass_below_abs < 0.05
    assert rm.w_median > 0.30


def test_quotient_tail_is_heavier_when_redundancy_concentrates_at_zero():
    """A1's heavy-tail half: the quotient's upper tail is heavier in the W -> 0 regime."""
    xi, yi = _independent()
    xc, yc = _correlated()
    indep = redundancy_mass(xi, yi, k_selected=10, random_state=0)
    corr = redundancy_mass(xc, yc, k_selected=10, random_state=0)
    assert indep.tail_slope < corr.tail_slope  # smaller exponent == heavier tail


def test_deterministic_given_random_state():
    X, y = _independent()
    a = redundancy_mass(X, y, k_selected=10, random_state=7)
    b = redundancy_mass(X, y, k_selected=10, random_state=7)
    assert a == b


def test_tail_slope_recovers_known_pareto_tail_index():
    """Hill-estimator correctness against a distribution with a KNOWN tail index.

    A classical Type-I Pareto(alpha) has tail index exactly alpha at every scale
    (it is an exact power law, not just asymptotically), so a correct Hill
    estimator applied to a large sample must recover alpha. Averaging many
    replicates isolates the estimator's systematic bias from its sampling noise.

    The buggy implementation used the smallest element OF THE SELECTED SLICE
    itself as the Hill threshold, injecting a spurious log(x/x) = 0 term into
    every replicate's log-ratio average. That drags the mean down and therefore
    pushes 1/mean UP -- a systematic upward bias in the exponent, i.e. the tail
    looks lighter (less heavy) than it really is. The tolerances below were set
    from the measured buggy-vs-fixed separation (buggy ~0.16-0.32 off, fixed
    ~0.07-0.15 off, noise ~0.01) so the buggy estimator fails this test and the
    corrected Hill estimator (threshold = the (k+1)-th order statistic, strictly
    outside the averaged top-k slice) passes it.
    """
    rng = np.random.default_rng(20260712)
    cases = [(1.5, 0.12), (3.0, 0.25)]
    for alpha, tol in cases:
        estimates = [_tail_slope(_pareto_sample(rng, alpha, n=200)) for _ in range(4000)]
        mean_estimate = float(np.mean(estimates))
        assert abs(mean_estimate - alpha) < tol, (
            f"alpha={alpha}: mean Hill estimate {mean_estimate:.4f} off by "
            f"{mean_estimate - alpha:+.4f}, exceeds tolerance {tol}"
        )


def test_scaled_threshold_removes_the_n_confound():
    """mass_below_abs is dominated by n; mass_below_scaled corrects for it.

    Same independent-column structure, very different n: the fixed-threshold
    statistic should swing wildly with n (this IS the confound found on the
    real datasets), while the noise-floor-scaled statistic should not.
    """
    x_small, y_small = _independent(n=100, seed=0)
    x_large, y_large = _independent(n=2000, seed=0)
    small = redundancy_mass(x_small, y_small, k_selected=10, random_state=0)
    large = redundancy_mass(x_large, y_large, k_selected=10, random_state=0)

    assert abs(small.mass_below_abs - large.mass_below_abs) > 0.3
    assert abs(small.mass_below_scaled - large.mass_below_scaled) < 0.3


def test_a1_w_median_scales_as_c_over_sqrt_n():
    """A1's first claim (paper Sec. 3 / App. C): median W-hat ~ c/sqrt(n), c ~ 0.8.

    Pinned via the coefficient of variation of the implied constant
    (c_implied = w_median * sqrt(n)) across a sweep of n -- not declared
    tautologically. If the scaling law did not hold, c_implied would drift with n
    and the CV would be large.
    """
    n_grid = [200, 500, 1000, 2000]
    seeds = [0, 1, 2]
    c_implied = []
    for n in n_grid:
        for seed in seeds:
            X, y = _synthetic_independent_columns(n=n, p=2000, seed=seed)
            rm = redundancy_mass(X, y, k_selected=10, random_state=seed)
            c_implied.append(rm.w_median * np.sqrt(n))
    c_implied = np.array(c_implied)
    c_hat = float(c_implied.mean())
    cv = float(c_implied.std() / c_hat)

    assert 0.6 < c_hat < 1.0, f"c_hat={c_hat:.4f} is not in the paper's claimed ballpark (~0.8)"
    assert cv < 0.05, f"CV={cv:.4f} too large -- w_median does not cleanly scale as c/sqrt(n)"


def test_a1_quotient_tail_is_heavy_and_finite_on_independent_columns():
    """A1's second claim (paper Sec. 3 / App. C): the quotient score rel/W has a
    heavy, finite-exponent upper tail on independent columns (alpha_hat ~ 3.8), and
    that tail is demonstrably heavier than on a dense low-rank (correlated) matrix
    of the same shape, where W stays bounded away from zero.
    """
    xi, yi = _synthetic_independent_columns(n=500, p=2000, seed=0)
    xc, yc = _correlated(n=500, p=2000, seed=0)
    indep = redundancy_mass(xi, yi, k_selected=10, random_state=0)
    corr = redundancy_mass(xc, yc, k_selected=10, random_state=0)

    assert np.isfinite(indep.tail_slope), "Hill exponent must be finite for a polynomial tail"
    assert indep.tail_slope < 5.0, (
        f"tail_slope={indep.tail_slope:.3f} heavier than the paper's claimed ballpark (~3.8)"
    )
    assert indep.tail_slope < corr.tail_slope, (
        "independent-column tail must be strictly heavier (smaller exponent) than the "
        "correlated-matrix tail"
    )
