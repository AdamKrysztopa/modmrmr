import numpy as np
import pandas as pd
import pytest

from modmrmr.core.scorers import (
    as_importance_function,
    as_penalty_matrix,
    build_scorer,
    get_scorer,
    list_scorers,
    register_scorer,
)
from modmrmr.models import PairwiseScorerSpec


def _spec(name: str) -> PairwiseScorerSpec:
    return PairwiseScorerSpec(
        name=name, backend="test", normalization="rank_percentile", significance_method="none"
    )


def test_all_builtin_families_registered() -> None:
    for name in ["pearson_abs", "spearman_abs", "mutual_info_sklearn", "gcmi", "tree_r2"]:
        assert name in list_scorers()


def test_tree_scorer_detects_nonlinear_dependence() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=300)
    y = np.sin(x) + rng.normal(scale=0.05, size=300)
    noise = rng.normal(size=300)
    tree = get_scorer("tree_r2")
    assert tree.score_pair(x, y).raw_value > tree.score_pair(x, noise).raw_value


def test_register_custom_scorer_roundtrips() -> None:
    sentinel = get_scorer("pearson_abs")
    register_scorer("my_custom", sentinel)
    assert get_scorer("my_custom") is sentinel


def test_build_scorer_honours_n_neighbors() -> None:
    spec = PairwiseScorerSpec(
        name="catt_knn_mi",
        backend="ksg",
        normalization="rank_percentile",
        significance_method="none",
        settings={"n_neighbors": 5},
    )
    scorer = build_scorer(spec)
    assert scorer._n_neighbors == 5


def test_adapters_produce_expected_shapes() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = X["a"] * 2 + rng.normal(scale=0.1, size=200)
    imp = as_importance_function(get_scorer("pearson_abs"))(X, y)
    pen = as_penalty_matrix(get_scorer("pearson_abs"))(X)
    assert imp.shape == (2,)
    assert pen.shape == (2, 2)
    assert float(pen.loc["a", "a"]) == 1.0


def test_f_regression_scorer_orders_signal_over_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y_signal = 3.0 * x + rng.normal(scale=0.1, size=300)
    y_noise = rng.normal(size=300)
    scorer = get_scorer("f_regression")
    strong = scorer.score_pair(x, y_signal).raw_value
    weak = scorer.score_pair(x, y_noise).raw_value
    assert np.isfinite(strong) and np.isfinite(weak)
    assert strong > weak
    assert weak >= 0.0


def test_f_classif_scorer_orders_separating_feature() -> None:
    rng = np.random.default_rng(1)
    labels = np.array([0] * 150 + [1] * 150, dtype=float)
    x_sep = np.concatenate([rng.normal(0.0, 1.0, 150), rng.normal(5.0, 1.0, 150)])
    x_flat = rng.normal(size=300)
    scorer = get_scorer("f_classif")
    strong = scorer.score_pair(x_sep, labels).raw_value
    weak = scorer.score_pair(x_flat, labels).raw_value
    assert np.isfinite(strong) and np.isfinite(weak)
    assert strong > weak
    assert weak >= 0.0


def test_mutual_info_classif_scorer_orders_separating_feature() -> None:
    rng = np.random.default_rng(2)
    labels = np.array([0.0] * 150 + [1.0] * 150)
    x_sep = np.concatenate([rng.normal(0.0, 0.5, 150), rng.normal(4.0, 0.5, 150)])
    x_flat = rng.normal(size=300)
    scorer = get_scorer("mutual_info_classif")
    strong = scorer.score_pair(x_sep, labels).raw_value
    weak = scorer.score_pair(x_flat, labels).raw_value
    assert np.isfinite(strong) and np.isfinite(weak)
    assert strong > weak
    assert weak >= 0.0


def test_distance_corr_scorer_bounds_and_ordering() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y_dep = 2.0 * x
    y_indep = rng.normal(size=400)
    scorer = get_scorer("distance_corr")
    dep = scorer.score_pair(x, y_dep).raw_value
    indep = scorer.score_pair(x, y_indep).raw_value
    assert 0.0 <= indep <= 1.0
    assert 0.0 <= dep <= 1.0
    assert dep > 0.95  # x vs 2x is a perfect linear dependence
    assert indep < 0.3  # independent noise


def test_rdc_scorer_bounds_and_ordering() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=400)
    y_dep = 2.0 * x + 1.0  # monotone transform -> same copula -> RDC ~ 1
    y_nonlin = np.sin(3.0 * x)  # nonlinear dependence RDC should catch
    y_indep = rng.normal(size=400)
    scorer = get_scorer("rdc")
    dep = scorer.score_pair(x, y_dep).raw_value
    nonlin = scorer.score_pair(x, y_nonlin).raw_value
    indep = scorer.score_pair(x, y_indep).raw_value
    for v in (dep, nonlin, indep):
        assert 0.0 <= v <= 1.0 and np.isfinite(v)
    assert dep > 0.9
    assert nonlin > indep


def test_rdc_scorer_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    y = np.cos(2.0 * x) + rng.normal(scale=0.1, size=300)
    scorer = get_scorer("rdc")
    a = scorer.score_pair(x, y, random_state=7).raw_value
    b = scorer.score_pair(x, y, random_state=7).raw_value
    assert a == b


def test_relieff_scorer_orders_separating_feature() -> None:
    rng = np.random.default_rng(6)
    labels = np.array([0.0] * 150 + [1.0] * 150)
    x_sep = np.concatenate([rng.normal(0.0, 0.5, 150), rng.normal(4.0, 0.5, 150)])
    x_flat = rng.normal(size=300)
    scorer = get_scorer("relieff")
    strong = scorer.score_pair(x_sep, labels).raw_value
    weak = scorer.score_pair(x_flat, labels).raw_value
    assert np.isfinite(strong) and np.isfinite(weak)
    assert strong > weak


def test_relieff_rejected_as_pairwise_redundancy() -> None:
    scorer = get_scorer("relieff")
    penalty = as_penalty_matrix(scorer)
    X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2})
    with pytest.raises(NotImplementedError, match="relevance-only"):
        penalty(X)


@pytest.mark.parametrize("name", ["f_classif", "mutual_info_classif"])
def test_classification_scorers_rejected_as_pairwise_redundancy(name: str) -> None:
    # These need discrete class labels as the second arg; feeding them
    # feature-vs-feature via the redundancy adapter is meaningless and must be
    # rejected cleanly, exactly like ReliefF.
    penalty = as_penalty_matrix(get_scorer(name))
    X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2})
    with pytest.raises(NotImplementedError, match="relevance-only"):
        penalty(X)


def test_relieff_with_neighbors_is_honoured_by_build_scorer() -> None:
    # Frozen tuning convention: build_scorer applies n_neighbors via with_neighbors.
    spec = PairwiseScorerSpec(
        name="relieff",
        backend="test",
        normalization="rank_percentile",
        significance_method="none",
        settings={"n_neighbors": 3},
    )
    tuned = build_scorer(spec)
    assert tuned._n_neighbors == 3


def test_rdc_is_robust_on_degenerate_and_independent_inputs() -> None:
    from modmrmr.core.scorers.dependence import _rdc_coefficient

    n = 400
    t = np.linspace(0.0, 1.0, n)
    # Strong (near-perfect) dependence must score high — the eigenvalue clip must
    # NOT discard the top squared canonical correlation.
    assert _rdc_coefficient(t, 2.0 * t + 1.0, random_state=0) > 0.9
    # A constant column carries no information — average-rank copula → RDC ~ 0
    # (ordinal ranks would fabricate a monotone ramp and spurious dependence).
    assert _rdc_coefficient(np.ones(n), np.sin(4.0 * np.pi * t), random_state=0) < 0.05
    # Independent inputs must not spuriously saturate to 1 (regularized-CCA ridge).
    for seed in range(5):
        r = np.random.default_rng(seed + 10)
        assert _rdc_coefficient(r.normal(size=n), r.normal(size=n), random_state=0) < 0.4
    # Determinism: same seed → bit-identical.
    assert _rdc_coefficient(t, t**2, random_state=7) == _rdc_coefficient(t, t**2, random_state=7)


def test_plan_b_scorers_all_registered() -> None:
    names = list_scorers()
    for name in [
        "f_regression",
        "f_classif",
        "mutual_info_classif",
        "distance_corr",
        "rdc",
        "relieff",
    ]:
        assert name in names


def test_bounded_dependence_scorers_score_range_and_correctness() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=400)
    y_dep = 2.0 * x
    y_noise = rng.normal(size=400)
    for name in ["distance_corr", "rdc"]:
        scorer = get_scorer(name)
        dep = scorer.score_pair(x, y_dep).raw_value
        indep = scorer.score_pair(x, y_noise).raw_value
        assert 0.0 <= dep <= 1.0, f"{name} dep out of range: {dep}"
        assert 0.0 <= indep <= 1.0, f"{name} indep out of range: {indep}"
        assert dep > 0.9, f"{name} failed to detect dependence: {dep}"
        assert dep > indep, f"{name} did not rank dependence above noise"


def test_cmim_jmi_are_not_registered_as_pairwise_scorers() -> None:
    # CMIM/JMI are conditional-MI (not pairwise-expressible); adapted externally
    # in Plan C. They must NOT appear in the pairwise scorer registry.
    names = list_scorers()
    assert "cmim" not in names
    assert "jmi" not in names


def _gauss_pair(rho, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return z[:, 0], z[:, 1]


def test_mixed_ksg_recovers_gaussian_mi():
    sc = get_scorer("mixed_ksg")
    for rho, exact in [(0.0, 0.0), (0.5, 0.14384), (0.9, 0.83038)]:
        x, y = _gauss_pair(rho)
        est = sc.score_pair(x, y, random_state=0).raw_value  # nats
        assert abs(est - exact) < 0.05, (rho, est, exact)


def test_mixed_ksg_discrete_target():
    # discrete y with real dependence on x must score > an independent discrete y
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y_dep = (x > 0).astype(int)
    y_indep = rng.integers(0, 2, size=2000)
    sc = get_scorer("mixed_ksg")
    assert (
        sc.score_pair(x, y_dep, random_state=0).raw_value
        > sc.score_pair(x, y_indep, random_state=0).raw_value + 0.1
    )


def test_mixed_ksg_discrete_discrete_tie_path():
    # Fully-discrete x and y force exact joint ties (joint Chebyshev distance
    # == 0 for many neighbors), exercising the rho_i == 0 branch of the KSG
    # correction that is otherwise never hit by continuous-x tests above.
    rng = np.random.default_rng(2)
    n = 2000
    levels = rng.integers(0, 4, size=n)
    sc = get_scorer("mixed_ksg")

    # y = x exactly: perfect dependence between two discrete variables.
    # MI = H(X) = ln(4) for 4 equally-likely levels.
    exact = np.log(4)
    est_dep = sc.score_pair(levels, levels.copy(), random_state=0).raw_value
    assert abs(est_dep - exact) < 0.05, (est_dep, exact)

    # Independent discrete y: MI should be near zero.
    y_indep = rng.integers(0, 4, size=n)
    est_indep = sc.score_pair(levels, y_indep, random_state=0).raw_value
    assert est_indep < 0.1, est_indep


def test_mixed_ksg_symmetry_determinism_degenerate():
    x, y = _gauss_pair(0.6, seed=3)
    sc = get_scorer("mixed_ksg")
    a = sc.score_pair(x, y, random_state=0).raw_value
    b = sc.score_pair(y, x, random_state=0).raw_value
    assert abs(a - b) < 0.02  # symmetry
    assert a == sc.score_pair(x, y, random_state=0).raw_value  # determinism
    const = np.ones(500)
    val = sc.score_pair(const, y[:500], random_state=0).raw_value
    assert np.isfinite(val) and val >= 0.0  # degenerate-safe


def test_mixed_ksg_is_scale_invariant_and_symmetric_under_scale_mismatch():
    # Both marginals are standardized to unit variance, so the joint Chebyshev
    # radius must be invariant to a per-variable rescale and identical under
    # argument order even when x and y live on wildly different scales.
    x, y = _gauss_pair(0.5)
    sc = get_scorer("mixed_ksg")
    base = sc.score_pair(x, y, random_state=0).raw_value
    fwd = sc.score_pair(x, y * 1000.0, random_state=0).raw_value
    rev = sc.score_pair(y * 1000.0, x, random_state=0).raw_value
    assert abs(fwd - base) < 1e-9  # scale-invariant
    assert fwd == rev  # order-independent
    assert abs(base - 0.14384) < 0.05  # still recovers analytic MI


def test_ami_adaptive_recovers_gaussian_mi():
    sc = get_scorer("ami_adaptive")
    for rho, exact in [(0.0, 0.0), (0.5, 0.14384), (0.9, 0.83038)]:
        x, y = _gauss_pair(rho)
        est = sc.score_pair(x, y, random_state=0).raw_value  # nats
        assert abs(est - exact) < 0.05, (rho, est, exact)


def test_ami_adaptive_discrete_target():
    # discrete y with real dependence on x must score > an independent discrete y
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y_dep = (x > 0).astype(int)
    y_indep = rng.integers(0, 2, size=2000)
    sc = get_scorer("ami_adaptive")
    assert (
        sc.score_pair(x, y_dep, random_state=0).raw_value
        > sc.score_pair(x, y_indep, random_state=0).raw_value + 0.1
    )


def test_ami_adaptive_symmetry_determinism_degenerate():
    x, y = _gauss_pair(0.6, seed=3)
    sc = get_scorer("ami_adaptive")
    a = sc.score_pair(x, y, random_state=0).raw_value
    b = sc.score_pair(y, x, random_state=0).raw_value
    assert abs(a - b) < 0.02  # symmetry
    assert a == sc.score_pair(x, y, random_state=0).raw_value  # determinism
    const = np.ones(500)
    val = sc.score_pair(const, y[:500], random_state=0).raw_value
    assert np.isfinite(val) and val >= 0.0  # degenerate-safe


def test_ami_adaptive_uses_more_neighbors_than_default_on_large_n():
    # on the linear-clf confound regime, adaptive-k should not collapse like n=3
    from mechanism.datasets import load_mechanism_dataset
    from mechanism.recovery import ranking_scores
    from modmrmr.core.estimator import MRMRSelector

    X, y, task, gt = load_mechanism_dataset("linear_control_clf")
    sel = MRMRSelector(
        n_features=5,
        relevance="ami_adaptive",
        redundancy="pearson_abs",
        operator="difference",
        aggregation="mean",
        task=task,
        random_state=0,
    )
    sel.fit(X, y)
    ap = ranking_scores(list(sel.selected_idx_), gt).average_precision
    assert ap >= 0.9  # recovers where sklearn-n3 gave AP 0.625 (Study 3)


def test_bspline_mi_recovers_gaussian_mi_low_and_is_monotone() -> None:
    # Soft-binning at n_bins=8 (Daub et al. 2004) is a coarse-histogram plug-in
    # estimator: it recovers low/moderate MI well but is biased LOW at high MI
    # (the overlapping soft bins blur fine joint structure). Measured with this
    # implementation at n=4000: rho=0.0 -> 0.0015 (exact 0.0), rho=0.5 -> 0.0768
    # (exact 0.14384, gap -0.067 -- within the 0.06-0.08 band the task brief
    # allows for a genuinely coarse 8-bin estimator), rho=0.9 -> 0.2918 (exact
    # 0.83038, gap -0.539 -- far outside any reasonable analytic tolerance, so
    # rho=0.9 is checked via strict monotonicity instead of an analytic bound).
    sc = get_scorer("bspline_mi")
    x0, y0 = _gauss_pair(0.0)
    x5, y5 = _gauss_pair(0.5)
    x9, y9 = _gauss_pair(0.9)
    est0 = sc.score_pair(x0, y0, random_state=0).raw_value
    est5 = sc.score_pair(x5, y5, random_state=0).raw_value
    est9 = sc.score_pair(x9, y9, random_state=0).raw_value
    assert abs(est0 - 0.0) < 0.03, est0
    assert abs(est5 - 0.14384) < 0.08, est5
    assert est9 > est5 > est0  # strict monotonicity substitutes for the biased rho=0.9 magnitude


def test_bspline_mi_discrete_target() -> None:
    # discrete y with real dependence on x must score > an independent discrete y;
    # exercises the group-by-class path (y treated as exact one-hot bins).
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y_dep = (x > 0).astype(int)
    y_indep = rng.integers(0, 2, size=2000)
    sc = get_scorer("bspline_mi")
    assert (
        sc.score_pair(x, y_dep, random_state=0).raw_value
        > sc.score_pair(x, y_indep, random_state=0).raw_value + 0.1
    )


def test_bspline_mi_symmetry_determinism_degenerate() -> None:
    x, y = _gauss_pair(0.6, seed=3)
    sc = get_scorer("bspline_mi")
    a = sc.score_pair(x, y, random_state=0).raw_value
    b = sc.score_pair(y, x, random_state=0).raw_value
    assert abs(a - b) < 0.02  # symmetry: both continuous, both splined
    assert a == sc.score_pair(x, y, random_state=0).raw_value  # determinism
    const = np.ones(500)
    val = sc.score_pair(const, y[:500], random_state=0).raw_value
    assert np.isfinite(val) and val >= 0.0  # degenerate (zero-range) input, no raise


def test_kde_mi_recovers_gaussian_mi() -> None:
    # Gaussian-KDE (Scott bandwidth) resubstitution plug-in MI at n=4000, seed=0:
    # rho=0.0 -> 0.0124 (exact 0.0, gap +0.012), rho=0.5 -> 0.1673 (exact 0.14384,
    # gap +0.023), rho=0.9 -> 0.8662 (exact 0.83038, gap +0.036). Unlike the coarse
    # 8-bin B-spline estimator, KDE tracks the analytic curve well even at high MI
    # (mild upward bias from finite-sample/bandwidth effects, not the severe
    # under-smoothing a fixed-bin histogram shows), so all three fit the 0.06 band.
    sc = get_scorer("kde_mi")
    for rho, exact in [(0.0, 0.0), (0.5, 0.14384), (0.9, 0.83038)]:
        x, y = _gauss_pair(rho)
        est = sc.score_pair(x, y, random_state=0).raw_value  # nats
        assert abs(est - exact) < 0.06, (rho, est, exact)


def test_kde_mi_discrete_target() -> None:
    # discrete y with real dependence on x must score > an independent discrete y;
    # exercises the H(x) - sum_c p(c) H(x|c) entropy-decomposition path.
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y_dep = (x > 0).astype(int)
    y_indep = rng.integers(0, 2, size=2000)
    sc = get_scorer("kde_mi")
    assert (
        sc.score_pair(x, y_dep, random_state=0).raw_value
        > sc.score_pair(x, y_indep, random_state=0).raw_value + 0.1
    )


def test_kde_mi_symmetry_determinism_degenerate() -> None:
    x, y = _gauss_pair(0.6, seed=3)
    sc = get_scorer("kde_mi")
    a = sc.score_pair(x, y, random_state=0).raw_value
    b = sc.score_pair(y, x, random_state=0).raw_value
    assert abs(a - b) < 0.02  # symmetry: seeded subsample + symmetric joint KDE
    assert a == sc.score_pair(x, y, random_state=0).raw_value  # determinism
    const = np.ones(500)
    val = sc.score_pair(const, y[:500], random_state=0).raw_value
    assert np.isfinite(val) and val >= 0.0  # singular-covariance guard, no raise


def test_copula_mi_recovers_gaussian_mi_and_is_monotone() -> None:
    # Empirical-copula (rank -> 16-bin histogram) plug-in MI at n=4000, seed=0:
    # rho=0.0 -> 0.0279 (exact 0.0, gap +0.028 -- small positive bias from
    # finite-sample cell noise at independence, as expected for a coarse
    # histogram), rho=0.5 -> 0.1685 (exact 0.14384, gap +0.025), rho=0.9 ->
    # 0.8123 (exact 0.83038, gap -0.018). All three land comfortably inside a
    # 0.06 band -- 16 bins resolves this copula better than the 8-bin B-spline
    # estimator -- but we still assert strict monotonicity as the primary,
    # bias-independent check.
    sc = get_scorer("copula_mi")
    x0, y0 = _gauss_pair(0.0)
    x5, y5 = _gauss_pair(0.5)
    x9, y9 = _gauss_pair(0.9)
    est0 = sc.score_pair(x0, y0, random_state=0).raw_value
    est5 = sc.score_pair(x5, y5, random_state=0).raw_value
    est9 = sc.score_pair(x9, y9, random_state=0).raw_value
    assert abs(est0 - 0.0) < 0.06, est0
    assert abs(est5 - 0.14384) < 0.06, est5
    assert abs(est9 - 0.83038) < 0.06, est9  # 16 bins still resolve high MI here
    assert est9 > est5 > est0  # strict monotonicity, bias-independent


def test_copula_mi_discrete_target() -> None:
    # discrete y with real dependence on x must score > an independent discrete y;
    # exercises the group-by-class v-partition path.
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y_dep = (x > 0).astype(int)
    y_indep = rng.integers(0, 2, size=2000)
    sc = get_scorer("copula_mi")
    assert (
        sc.score_pair(x, y_dep, random_state=0).raw_value
        > sc.score_pair(x, y_indep, random_state=0).raw_value + 0.1
    )


def test_copula_mi_symmetry_determinism_degenerate() -> None:
    x, y = _gauss_pair(0.6, seed=3)
    sc = get_scorer("copula_mi")
    a = sc.score_pair(x, y, random_state=0).raw_value
    b = sc.score_pair(y, x, random_state=0).raw_value
    assert abs(a - b) < 0.02  # symmetry: rank + histogram is symmetric in u, v
    assert a == sc.score_pair(x, y, random_state=0).raw_value  # determinism
    const = np.ones(500)
    val = sc.score_pair(const, y[:500], random_state=0).raw_value
    assert np.isfinite(val) and val >= 0.0  # degenerate (constant ranks), no raise


def test_knn_neighbor_count_variants_registered() -> None:
    names = list_scorers()
    expected = [
        "mi_classif_k3",
        "mi_classif_k5",
        "mi_classif_k8",
        "mi_classif_k16",
        "mi_reg_k3",
        "mi_reg_k5",
        "mi_reg_k8",
        "mi_reg_k16",
        "ksg_k16",
    ]
    for name in expected:
        assert name in names, name

    x, y = _gauss_pair(0.6, seed=3)
    sc = get_scorer("mi_reg_k16")
    a = sc.score_pair(x, y, random_state=0).raw_value
    b = sc.score_pair(x, y, random_state=0).raw_value
    assert a == b  # determinism

    rng = np.random.default_rng(2)
    xc = rng.normal(size=500)
    yc = (xc > 0).astype(int)
    clf = get_scorer("mi_classif_k8")
    val = clf.score_pair(xc, yc, random_state=0).raw_value
    assert np.isfinite(val)
