import pytest

from mechanism.ground_truth import GroundTruth
from mechanism.recovery import RankingScore, RecoveryScore, ranking_scores, recovery


def _gt() -> GroundTruth:
    # 6 columns: 0,1 informative; group (2,3) codependent; 4,5 noise
    return GroundTruth(
        informative=(0, 1),
        codependent=((2, 3),),
        noise=(4, 5),
        dependence="nonlinear",
        n_features=6,
    )


def test_perfect_informative_pick():
    gt = _gt()
    score = recovery(list(gt.informative), gt)
    assert score == RecoveryScore(
        precision=1.0, recall=1.0, f1=1.0, redundancy_rate=0.0, noise_rate=0.0
    )


def test_all_noise_pick():
    gt = _gt()
    score = recovery(list(gt.noise), gt)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.redundancy_rate == 0.0
    assert score.noise_rate == 1.0


def test_both_codependent_group_members_selected():
    gt = _gt()
    score = recovery([2, 3], gt)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.redundancy_rate == 0.5
    assert score.noise_rate == 0.0


def test_empty_selected_is_all_zero_no_zero_division():
    gt = _gt()
    score = recovery([], gt)
    assert score == RecoveryScore(
        precision=0.0, recall=0.0, f1=0.0, redundancy_rate=0.0, noise_rate=0.0
    )


def test_mixed_case_hand_computed():
    # 8 columns: 0,1 informative; groups (2,3) and (4,5) codependent; 6,7 noise.
    gt = GroundTruth(
        informative=(0, 1),
        codependent=((2, 3), (4, 5)),
        noise=(6, 7),
        dependence="mixed",
        n_features=8,
    )
    # order matters for redundancy_rate: 2 is the group-0 representative, 3 is
    # its redundant duplicate; 4 is the group-1 representative (first pick from
    # that group so far).
    selected = [0, 2, 6, 3, 4]
    score = recovery(selected, gt)

    # hits: only idx 0 is informative -> precision = 1/5, recall = 1/2
    assert score.precision == pytest.approx(0.2)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(2 * 0.2 * 0.5 / (0.2 + 0.5))
    # redundant: only idx 3 (second pick from group 0) -> 1/5
    assert score.redundancy_rate == pytest.approx(0.2)
    # noise: only idx 6 -> 1/5
    assert score.noise_rate == pytest.approx(0.2)


def test_recovery_score_is_frozen():
    score = recovery([0], _gt())
    with pytest.raises(Exception):  # noqa: B017 - dataclass(frozen=True) raises FrozenInstanceError
        score.precision = 0.5  # type: ignore[misc]


def test_ranking_scores_perfect_order_gives_perfect_average_precision():
    gt = _gt()
    score = ranking_scores([0, 1, 2, 3, 4, 5], gt)
    assert isinstance(score, RankingScore)
    assert score.average_precision == pytest.approx(1.0)
    assert score.roc_auc == pytest.approx(1.0)


def test_ranking_scores_worst_order_gives_low_average_precision():
    gt = _gt()
    score = ranking_scores([4, 5, 2, 3, 0, 1], gt)
    assert score.average_precision < 0.5


def test_ranking_scores_empty_informative_returns_zero_no_crash():
    gt = GroundTruth(
        informative=(),
        codependent=((0, 1),),
        noise=(2, 3, 4, 5),
        dependence="linear",
        n_features=6,
    )
    score = ranking_scores([0, 1, 2, 3, 4, 5], gt)
    assert score == RankingScore(average_precision=0.0, roc_auc=0.0)


def test_ranking_scores_all_columns_informative_returns_zero_no_crash():
    gt = GroundTruth(
        informative=(0, 1, 2),
        codependent=(),
        noise=(),
        dependence="linear",
        n_features=3,
    )
    score = ranking_scores([0, 1, 2], gt)
    assert score == RankingScore(average_precision=0.0, roc_auc=0.0)


def test_ranking_scores_is_frozen():
    score = ranking_scores([0, 1], _gt())
    with pytest.raises(Exception):  # noqa: B017 - dataclass(frozen=True) raises FrozenInstanceError
        score.average_precision = 0.5  # type: ignore[misc]


def test_ranking_scores_deterministic():
    gt = _gt()
    s1 = ranking_scores([0, 1, 2, 3, 4, 5], gt)
    s2 = ranking_scores([0, 1, 2, 3, 4, 5], gt)
    assert s1 == s2


def test_ranking_scores_partial_selection_unselected_are_never_selected():
    # only informative selected; the rest never appear in selection_order
    gt = _gt()
    score = ranking_scores([0, 1], gt)
    assert score.average_precision == pytest.approx(1.0)


def test_ranking_scores_rejects_duplicate_indices():
    gt = _gt()
    with pytest.raises(ValueError, match="distinct"):
        ranking_scores([0, 0, 1], gt)


def test_ranking_scores_rejects_out_of_range_index():
    gt = _gt()
    with pytest.raises(ValueError, match="range"):
        ranking_scores([0, 1, 99], gt)
