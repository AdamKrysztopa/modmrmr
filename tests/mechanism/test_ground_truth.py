import pytest

from mechanism.ground_truth import GroundTruth


def _gt() -> GroundTruth:
    # 6 columns: 0,1 informative; group (2,3) codependent; 4,5 noise
    return GroundTruth(
        informative=(0, 1),
        codependent=((2, 3),),
        noise=(4, 5),
        dependence="nonlinear",
        n_features=6,
    )


def test_relevant_columns_is_informative_plus_codependent():
    assert _gt().relevant_columns == frozenset({0, 1, 2, 3})


def test_codependent_group_lookup():
    gt = _gt()
    assert gt.codependent_group_of(2) == 0
    assert gt.codependent_group_of(3) == 0
    assert gt.codependent_group_of(0) is None


def test_indices_must_partition_all_columns():
    with pytest.raises(ValueError, match="partition"):
        GroundTruth(informative=(0,), codependent=(), noise=(0,), dependence="linear", n_features=2)


def test_dependence_must_be_valid():
    with pytest.raises(ValueError, match="dependence"):
        GroundTruth(informative=(0,), codependent=(), noise=(1,), dependence="wat", n_features=2)


def test_is_frozen():
    with pytest.raises(Exception):  # noqa: B017 - dataclass(frozen=True) raises FrozenInstanceError
        _gt().informative = (9,)  # type: ignore[misc]
