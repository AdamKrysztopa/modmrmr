import dcor
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from mechanism.ground_truth import GroundTruth
from mechanism.synthetic import (
    _assemble,
    make_annulus,
    make_checkerboard,
    make_conditional_redundancy,
    make_heteroscedastic,
    make_linear_control_reg,
    make_max_of,
    make_mixed,
    make_multiplicative_interaction,
    make_nonlin_redundancy_clf,
    make_nonlin_redundancy_reg,
    make_parabola,
    make_polynomial_sign,
    make_quotient_trap_clf,
    make_quotient_trap_reg,
    make_radial,
    make_ratio,
    make_redundant_blocks_clf,
    make_redundant_blocks_reg,
    make_saturation,
    make_sinc_2d,
    make_sine,
    make_threshold_and,
    make_xor,
)


def test_assemble_shuffles_and_records_roles():
    rng = np.random.default_rng(0)
    cols = [np.arange(5.0), np.arange(5.0) + 10, np.arange(5.0) + 20, np.arange(5.0) + 30]
    roles = ["info", ("codep", 0), "noise", ("codep", 0)]
    y = np.array([0, 1, 0, 1, 0])
    X, ys, gt = _assemble(rng, cols, roles, y, "nonlinear")
    assert isinstance(X, pd.DataFrame) and isinstance(ys, pd.Series)
    assert X.shape == (5, 4)
    assert list(X.columns) == ["f0", "f1", "f2", "f3"]
    assert isinstance(gt, GroundTruth)
    assert len(gt.informative) == 1
    assert len(gt.codependent) == 1 and len(gt.codependent[0]) == 2
    assert len(gt.noise) == 1
    # the informative column's data survived to its recorded position
    info_pos = gt.informative[0]
    assert np.allclose(X.iloc[:, info_pos].to_numpy(), np.arange(5.0))


def test_assemble_is_deterministic():
    def build(seed):
        rng = np.random.default_rng(seed)
        cols = [np.arange(5.0) + 10 * i for i in range(3)]
        return _assemble(rng, cols, ["info", ("codep", 0), "noise"], np.zeros(5), "mixed")

    X1, _, g1 = build(7)
    X2, _, g2 = build(7)
    pd.testing.assert_frame_equal(X1, X2)
    assert g1 == g2


def _abs_pearson(a, b):
    return abs(pearsonr(a, b)[0])


def test_parabola_linear_blind_nonlinear_sees_relevance():
    X, y, gt = make_parabola(seed=0)
    info = X.iloc[:, gt.informative[0]].to_numpy()
    # linear correlation with y is ~0 (symmetric parabola) ...
    assert _abs_pearson(info, y.to_numpy()) < 0.1
    # ... but mutual information is clearly positive.
    mi = mutual_info_regression(info.reshape(-1, 1), y.to_numpy(), random_state=0)[0]
    assert mi > 0.1


def test_radial_marginal_mi_positive_pearson_zero():
    X, y, gt = make_radial(seed=0)
    info = X.iloc[:, gt.informative[0]].to_numpy()
    assert _abs_pearson(info, y.to_numpy()) < 0.1
    mi = mutual_info_regression(info.reshape(-1, 1), y.to_numpy().astype(float), random_state=0)[0]
    assert mi > 0.05


def test_sine_nonlinear_redundancy_twin_is_hidden_from_pearson():
    X, y, gt = make_sine(seed=0)
    info_pos = gt.informative[0]
    # the codependent group holds a NONLINEAR twin: dcor high, pearson low.
    codep_positions = [p for grp in gt.codependent for p in grp]
    x0 = X.iloc[:, info_pos].to_numpy()
    # at least one codependent twin is nonlinearly redundant with x0
    nonlinear_twin = max(
        codep_positions,
        key=lambda p: (
            dcor.distance_correlation(x0, X.iloc[:, p].to_numpy())
            - _abs_pearson(x0, X.iloc[:, p].to_numpy())
        ),
    )
    tw = X.iloc[:, nonlinear_twin].to_numpy()
    assert _abs_pearson(x0, tw) < 0.2
    assert dcor.distance_correlation(x0, tw) > 0.5


def test_generators_are_deterministic():
    for gen in (make_parabola, make_radial, make_sine):
        X1, y1, g1 = gen(seed=3)
        X2, y2, g2 = gen(seed=3)
        pd.testing.assert_frame_equal(X1, X2)
        assert g1 == g2


def test_nonlinear_redundancy_twin_invisible_to_pearson_visible_to_dcor():
    for gen in (make_nonlin_redundancy_clf, make_nonlin_redundancy_reg):
        X, y, gt = gen(seed=0)
        x0 = X.iloc[:, gt.informative[0]].to_numpy()
        twin_pos = gt.codependent[0][0]
        twin = X.iloc[:, twin_pos].to_numpy()
        assert _abs_pearson(x0, twin) < 0.2  # linear redundancy blind
        assert dcor.distance_correlation(x0, twin) > 0.5  # nonlinear sees it
        # relevance IS linearly detectable here (unlike the parabola set)
        assert _abs_pearson(x0, y.to_numpy().astype(float)) > 0.3


def test_linear_control_relevance_is_linearly_detectable():
    X, y, gt = make_linear_control_reg(seed=0)
    x0 = X.iloc[:, gt.informative[0]].to_numpy()
    assert _abs_pearson(x0, y.to_numpy().astype(float)) > 0.3  # linear measure succeeds
    assert gt.dependence == "linear"


def test_mixed_has_both_regimes_and_partitions():
    X, y, gt = make_mixed(seed=0)
    assert gt.dependence == "mixed"
    assert len(gt.informative) >= 2 and len(gt.codependent) >= 1 and len(gt.noise) >= 1
    # partition invariant already enforced by GroundTruth.__post_init__
    assert X.shape[1] == gt.n_features


def test_xor_is_marginally_invisible_to_all_pairwise_measures():
    X, y, gt = make_xor(seed=0)
    x0 = X.iloc[:, gt.informative[0]].to_numpy()
    yv = y.to_numpy().astype(float)
    assert _abs_pearson(x0, yv) < 0.1  # linear blind
    mi = mutual_info_regression(x0.reshape(-1, 1), yv, random_state=0)[0]
    assert mi < 0.05  # AND univariate MI blind — needs a conditional/interaction method


# --------------------------------------------------------------------------
# redundant_blocks — the redundancy trap
# --------------------------------------------------------------------------


def test_redundant_blocks_partition_and_structure():
    for gen in (make_redundant_blocks_clf, make_redundant_blocks_reg):
        X, y, gt = gen(seed=0)
        assert gt.n_features == 14
        assert X.shape == (1200, 14)
        assert len(gt.informative) == 3
        assert len(gt.codependent) == 2
        assert sorted(len(g) for g in gt.codependent) == [2, 3]
        assert len(gt.noise) == 6
        # determinism
        X2, y2, gt2 = gen(seed=0)
        pd.testing.assert_frame_equal(X, X2)
        pd.testing.assert_series_equal(y, y2)
        assert gt == gt2


def test_redundant_blocks_twins_are_pearson_redundant():
    for gen in (make_redundant_blocks_clf, make_redundant_blocks_reg):
        X, y, gt = gen(seed=0)
        for group in gt.codependent:
            # every twin in the group is a strongly linear-redundant duplicate
            # of *some* informative signal (we don't know which a priori, so
            # check against whichever informative column correlates best).
            for pos in group:
                twin = X.iloc[:, pos].to_numpy()
                best = max(
                    _abs_pearson(twin, X.iloc[:, info_pos].to_numpy())
                    for info_pos in gt.informative
                )
                assert best > 0.9


def test_redundant_blocks_relevance_is_linear():
    for gen in (make_redundant_blocks_clf, make_redundant_blocks_reg):
        X, y, gt = gen(seed=0)
        yv = y.to_numpy().astype(float)
        rel = {pos: _abs_pearson(X.iloc[:, pos].to_numpy(), yv) for pos in gt.informative}
        strong_pos = max(rel, key=rel.get)
        weak_pos = min(rel, key=rel.get)
        assert rel[strong_pos] > rel[weak_pos] > 0.05


def test_redundant_blocks_TRAP():
    """The key property: a strong-signal twin outranks the weak true signal in
    y-relevance. This is what makes a redundancy-blind operator fail."""
    for gen in (make_redundant_blocks_clf, make_redundant_blocks_reg):
        X, y, gt = gen(seed=0)
        yv = y.to_numpy().astype(float)
        rel = {pos: _abs_pearson(X.iloc[:, pos].to_numpy(), yv) for pos in gt.informative}
        weak_pos = min(rel, key=rel.get)
        weak_rel = rel[weak_pos]
        # the largest group (3 twins) belongs to the strong signal
        strong_group = max(gt.codependent, key=len)
        twin_rels = [_abs_pearson(X.iloc[:, pos].to_numpy(), yv) for pos in strong_group]
        assert max(twin_rels) > weak_rel


def test_redundant_blocks_separates_operators():
    """ACCEPTANCE TEST: at fixed aggregation, the multiplicative operator must
    arbitrate the redundancy trap better than the difference operator — this is
    the entire reason `redundant_blocks_reg` exists. Do NOT weaken this
    assertion; strengthen the trap in the generator instead.

    ``relevance="f_regression"`` matches the estimator's own regression default
    (``DEFAULT_MEASURES``) and is deliberately on an *unbounded* scale versus
    ``redundancy="pearson_abs"``'s ``[0, 1]`` scale. That scale mismatch is what
    exposes the operator difference here: ``difference`` subtracts a bounded
    redundancy value from an unbounded relevance value, so the penalty is
    negligible and it degenerates into picking almost purely by raw relevance
    — grabbing a second twin from the strong signal's group. ``multiplicative``
    applies a *proportional* ``(1 - redundancy)`` penalty, which is scale-free
    and still suppresses the duplicate twin regardless of the relevance
    measure's absolute scale.
    """
    from mechanism.recovery import recovery
    from modmrmr.core.estimator import MRMRSelector

    seeds = (0, 1, 2)
    diff_scores, mult_scores = [], []
    for seed in seeds:
        X, y, gt = make_redundant_blocks_reg(seed=seed)
        common = dict(
            n_features=3,
            relevance="f_regression",
            redundancy="pearson_abs",
            aggregation="mean",
            task="regression",
            random_state=seed,
        )
        diff = MRMRSelector(operator="difference", **common).fit(X, y)
        mult = MRMRSelector(operator="multiplicative", **common).fit(X, y)
        diff_scores.append(recovery(diff.selected_idx_, gt))
        mult_scores.append(recovery(mult.selected_idx_, gt))

    mean_recall_diff = np.mean([s.recall for s in diff_scores])
    mean_recall_mult = np.mean([s.recall for s in mult_scores])
    mean_redund_diff = np.mean([s.redundancy_rate for s in diff_scores])
    mean_redund_mult = np.mean([s.redundancy_rate for s in mult_scores])

    assert mean_recall_mult >= mean_recall_diff
    assert mean_redund_mult <= mean_redund_diff
    assert (mean_recall_mult - mean_recall_diff > 0.1) or (
        mean_redund_diff - mean_redund_mult > 0.1
    )


# --------------------------------------------------------------------------
# quotient_trap — the W->0 crowning-trap golden datasets
# --------------------------------------------------------------------------


def test_quotient_trap_partition_and_structure():
    for gen in (make_quotient_trap_reg, make_quotient_trap_clf):
        X, y, gt = gen(seed=0)
        assert gt.n_features == 13
        assert X.shape == (1200, 13)
        assert len(gt.informative) == 3
        assert len(gt.codependent) == 1
        assert len(gt.codependent[0]) == 2
        assert len(gt.noise) == 8
        assert gt.dependence == "linear"
        # determinism
        X2, y2, gt2 = gen(seed=0)
        pd.testing.assert_frame_equal(X, X2)
        pd.testing.assert_series_equal(y, y2)
        assert gt == gt2


def test_quotient_trap_twins_are_pearson_redundant_with_an_informative_signal():
    for gen in (make_quotient_trap_reg, make_quotient_trap_clf):
        X, y, gt = gen(seed=0)
        for group in gt.codependent:
            for pos in group:
                twin = X.iloc[:, pos].to_numpy()
                best = max(
                    _abs_pearson(twin, X.iloc[:, info_pos].to_numpy())
                    for info_pos in gt.informative
                )
                assert best > 0.9


def test_quotient_trap_distractors_are_independent_of_y():
    for gen in (make_quotient_trap_reg, make_quotient_trap_clf):
        X, y, gt = gen(seed=0)
        yv = y.to_numpy().astype(float)
        for pos in gt.noise:
            distractor = X.iloc[:, pos].to_numpy()
            assert _abs_pearson(distractor, yv) < 0.15


def test_quotient_trap_clf_is_balanced():
    X, y, gt = make_quotient_trap_clf(seed=0)
    yv = y.to_numpy()
    assert set(np.unique(yv)) <= {0, 1}
    frac_positive = yv.mean()
    assert 0.35 < frac_positive < 0.65


# --------------------------------------------------------------------------
# Extended nonlinear golden suite — 11 mechanistically distinct generators.
# --------------------------------------------------------------------------

_NONLINEAR_GENERATORS = [
    make_checkerboard,
    make_annulus,
    make_multiplicative_interaction,
    make_threshold_and,
    make_saturation,
    make_ratio,
    make_max_of,
    make_heteroscedastic,
    make_sinc_2d,
    make_polynomial_sign,
    make_conditional_redundancy,
]

_CLASSIFICATION_GENERATORS = [
    make_checkerboard,
    make_annulus,
    make_threshold_and,
    make_polynomial_sign,
    make_conditional_redundancy,
]


@pytest.mark.parametrize("gen", _NONLINEAR_GENERATORS)
def test_nonlinear_generator_shapes_and_partition(gen):
    X, y, gt = gen(seed=0, n=600)
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
    # GroundTruth.__post_init__ already enforces the partition invariant.
    assert X.shape == (600, gt.n_features)
    assert len(y) == 600
    assert list(X.columns) == [f"f{i}" for i in range(gt.n_features)]
    assert gt.dependence == "nonlinear"
    assert len(gt.informative) >= 1
    assert len(gt.noise) >= 1


@pytest.mark.parametrize("gen", _NONLINEAR_GENERATORS)
def test_nonlinear_generator_is_deterministic(gen):
    X1, y1, g1 = gen(seed=5)
    X2, y2, g2 = gen(seed=5)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    assert g1 == g2


@pytest.mark.parametrize("gen", _NONLINEAR_GENERATORS)
def test_nonlinear_generator_relevance_is_linear_blind(gen):
    # Every informative feature is (near-)invisible to Pearson: the whole point of
    # the nonlinear suite is that linear correlation cannot see the signal.
    X, y, gt = gen(seed=0)
    yv = y.to_numpy().astype(float)
    for pos in gt.informative:
        assert _abs_pearson(X.iloc[:, pos].to_numpy(), yv) < 0.15


@pytest.mark.parametrize("gen", _CLASSIFICATION_GENERATORS)
def test_nonlinear_classification_generators_are_balanced(gen):
    X, y, gt = gen(seed=0)
    yv = y.to_numpy()
    assert set(np.unique(yv)) <= {0, 1}
    assert 0.35 < yv.mean() < 0.65


def test_multiplicative_interaction_only_via_product():
    # y = x0*x1: marginal MI is positive (each factor sets the conditional spread)
    # while Pearson stays ~0.
    X, y, gt = make_multiplicative_interaction(seed=0)
    yv = y.to_numpy().astype(float)
    for pos in gt.informative:
        xi = X.iloc[:, pos].to_numpy()
        assert _abs_pearson(xi, yv) < 0.1
        assert mutual_info_regression(xi.reshape(-1, 1), yv, random_state=0)[0] > 0.1


def test_saturation_has_nonlinear_codependent_twin():
    X, y, gt = make_saturation(seed=0)
    assert len(gt.codependent) == 1 and len(gt.codependent[0]) == 1
    x0 = X.iloc[:, gt.informative[0]].to_numpy()
    twin = X.iloc[:, gt.codependent[0][0]].to_numpy()
    assert _abs_pearson(x0, twin) < 0.2  # fold twin is linear-blind ...
    assert dcor.distance_correlation(x0, twin) > 0.5  # ... but dcor-visible


def test_conditional_redundancy_exercises_a_codependent_group():
    X, y, gt = make_conditional_redundancy(seed=0)
    assert len(gt.codependent) == 1 and len(gt.codependent[0]) == 2
    infos = [X.iloc[:, p].to_numpy() for p in gt.informative]
    for pos in gt.codependent[0]:
        twin = X.iloc[:, pos].to_numpy()
        # linear-blind against every informative feature ...
        assert all(_abs_pearson(info, twin) < 0.2 for info in infos)
        # ... but nonlinearly redundant with (at least) one of them.
        assert max(dcor.distance_correlation(info, twin) for info in infos) > 0.5


def test_heteroscedastic_drives_variance_not_mean():
    # Correlation is blind (y symmetric given x0) but the conditional spread of y
    # grows with |x0| — high-|x0| samples have visibly larger |y|.
    X, y, gt = make_heteroscedastic(seed=0)
    yv = y.to_numpy().astype(float)
    x0 = X.iloc[:, gt.informative[0]].to_numpy()
    assert _abs_pearson(x0, yv) < 0.1
    hi = np.abs(yv)[np.abs(x0) > np.quantile(np.abs(x0), 0.75)].mean()
    lo = np.abs(yv)[np.abs(x0) < np.quantile(np.abs(x0), 0.25)].mean()
    assert hi > lo
