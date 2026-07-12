import importlib.util

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_friedman1

from benchmarks.cells import (
    BASELINES,
    CELLS,
    MRMRCellAdapter,
    SelectKBestAdapter,
    SelectorAdapter,
    SkfeatureBaseline,
    get_adapter,
    list_cells,
)


def _baseline_available(name: str) -> bool:
    """Whether the optional dependency backing a baseline is importable."""
    if name in {"cmim", "jmi", "cife", "icap", "disr"}:
        return SkfeatureBaseline.available()
    if name == "mrmr_smazzanti":
        return importlib.util.find_spec("mrmr") is not None
    return True  # skb_*/relieff/rfe rely on core deps (sklearn, skrebate)


def _clf_frame():
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=5, n_redundant=5, random_state=0
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]), pd.Series(y)


def _reg_frame():
    X, y = make_friedman1(n_samples=200, n_features=10, noise=0.1, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


def test_adapter_conforms_to_protocol():
    adapter = SelectKBestAdapter(name="skb_f", score="f")
    assert isinstance(adapter, SelectorAdapter)
    assert adapter.name == "skb_f"


def test_select_returns_k_distinct_valid_indices_classification():
    X, y = _clf_frame()
    adapter = SelectKBestAdapter(name="skb_f", score="f")
    idx = adapter.select(X, y, "classification", k=5)
    assert isinstance(idx, list)
    assert len(idx) == 5
    assert len(set(idx)) == 5
    assert all(0 <= i < X.shape[1] for i in idx)


def test_select_returns_k_indices_regression_mutual_info():
    X, y = _reg_frame()
    adapter = SelectKBestAdapter(name="skb_mi", score="mutual_info")
    idx = adapter.select(X, y, "regression", k=3)
    assert len(idx) == 3
    assert all(isinstance(i, int) for i in idx)


def test_select_caps_k_at_n_features():
    X, y = _reg_frame()
    adapter = SelectKBestAdapter(name="skb_f", score="f")
    idx = adapter.select(X, y, "regression", k=999)
    assert len(idx) == X.shape[1]


def test_cells_registry_contains_named_family_presets():
    for name in ["MID", "MIQ", "MIFS", "FCD", "FCQ", "ModMRMR"]:
        assert name in CELLS
        assert isinstance(CELLS[name], MRMRCellAdapter)


def test_literal_information_theoretic_cells_use_mi_redundancy():
    # MID/MIQ are the literal Peng-2005 mRMR criteria (MI relevance AND MI
    # redundancy), not the MI-relevance x linear-correlation-redundancy hybrid.
    # MIFS (Battiti 1994) is the beta=1 unweighted-sum form.
    assert CELLS["MID"].redundancy == "mutual_info_sklearn"
    assert CELLS["MID"].operator == "difference" and CELLS["MID"].aggregation == "mean"
    assert CELLS["MIQ"].redundancy == "mutual_info_sklearn"
    assert CELLS["MIQ"].operator == "quotient"
    assert CELLS["MIFS"].operator == "difference" and CELLS["MIFS"].aggregation == "sum"


def test_modmrmr_cell_uses_multiplicative_max():
    cell = CELLS["ModMRMR"]
    assert cell.operator == "multiplicative"
    assert cell.aggregation == "max"


def test_baselines_registry_contains_controls():
    for name in ["skb_f", "skb_mi", "mrmr_smazzanti", "relieff", "rfe"]:
        assert name in BASELINES


@pytest.mark.parametrize(
    "name",
    ["skb_f", "skb_mi", "mrmr_smazzanti", "relieff", "rfe", "cmim", "jmi", "cife", "icap", "disr"],
)
def test_every_baseline_selects_k_valid_indices_classification(name):
    # Behavioral smoke test: actually invoke .select() on each baseline (the
    # external-lib adapters had zero call-path coverage, which let a broken
    # skfeature return-signature ship). Gate only on optional-dep availability.
    if not _baseline_available(name):
        pytest.skip(f"optional dependency for {name!r} not installed")
    X, y = _clf_frame()
    idx = get_adapter(name).select(X, y, "classification", k=5)
    assert isinstance(idx, list)
    assert len(idx) == 5
    assert len(set(idx)) == 5
    assert all(0 <= i < X.shape[1] for i in idx)


def test_skb_mi_baseline_is_deterministic():
    # mutual_info_* must be seeded so the baseline (and the stability bootstrap
    # that refits it) is reproducible regardless of global RNG state.
    X, y = _clf_frame()
    adapter = get_adapter("skb_mi")
    import numpy as np

    np.random.seed(1)
    first = adapter.select(X, y, "classification", k=5)
    np.random.seed(999)
    second = adapter.select(X, y, "classification", k=5)
    assert first == second


def test_skfeature_baseline_rejects_regression():
    if not SkfeatureBaseline.available():
        pytest.skip("skfeature not installed")
    X, y = _reg_frame()
    with pytest.raises(ValueError, match="classification-only"):
        get_adapter("cmim").select(X, y, "regression", k=3)


def test_get_adapter_and_list_cells_span_both_registries():
    names = set(list_cells())
    assert {"MID", "skb_f"} <= names
    assert get_adapter("skb_f").name == "skb_f"
    with pytest.raises(KeyError):
        get_adapter("nope")


def test_mrmr_cell_relevance_resolution_by_task():
    cell = CELLS["MID"]
    assert cell.relevance_family == "mi"
    assert cell._resolve_relevance("classification") == "mutual_info_classif"
    assert cell._resolve_relevance("regression") == "mutual_info_sklearn"
    fcd = CELLS["FCD"]
    assert fcd._resolve_relevance("classification") == "f_classif"
    assert fcd._resolve_relevance("regression") == "f_regression"


@pytest.mark.skipif(
    importlib.util.find_spec("modmrmr.core.estimator") is None,
    reason="Plan A MRMRSelector not yet merged",
)
def test_mrmr_cell_select_end_to_end_when_plan_a_present():
    try:
        from modmrmr.core.estimator import MRMRSelector  # noqa: F401
    except Exception:
        pytest.skip("MRMRSelector import failed; Plan A incomplete")
    X, y = make_classification(n_samples=150, n_features=20, n_informative=5, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    idx = CELLS["MID"].select(X, pd.Series(y), "classification", k=5)
    assert len(idx) == 5
