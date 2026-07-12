"""Golden-master parity: MRMRSelector presets reproduce legacy MRMR/ModMRMR."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from modmrmr.core.estimator import MRMR, ModMRMR, MRMRSelector, pearson_corr


def _dataset(n: int = 600):
    """The exact golden dataset from tests/test_regression.py."""
    rng = np.random.default_rng(0)
    signal_a = rng.normal(size=n)
    signal_b = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "relevant_a": signal_a + rng.normal(scale=0.1, size=n),
            "duplicate_a": signal_a + rng.normal(scale=0.1, size=n),
            "relevant_b": signal_b + rng.normal(scale=0.1, size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = signal_a + signal_b
    return X, y


def _legacy_measures(**kwargs) -> MRMRSelector:
    return MRMRSelector(
        relevance=mutual_info_regression,
        redundancy=pearson_corr,
        task="regression",
        random_state=0,
        **kwargs,
    )


def test_quotient_mean_reproduces_legacy_mrmr() -> None:
    X, y = _dataset()
    legacy = MRMR(n_features=3, random_state=0).fit(X, y)
    engine = _legacy_measures(n_features=3, operator="quotient", aggregation="mean").fit(X, y)
    assert list(engine.selected_features_) == list(legacy.selected_features_)
    assert engine.selected_idx_ == legacy.selected_idx_


def test_multiplicative_max_reproduces_legacy_modmrmr() -> None:
    X, y = _dataset()
    legacy = ModMRMR(n_features=3, random_state=0).fit(X, y)
    engine = _legacy_measures(n_features=3, operator="multiplicative", aggregation="max").fit(X, y)
    assert list(engine.selected_features_) == list(legacy.selected_features_)
    assert engine.selected_idx_ == legacy.selected_idx_


def test_legacy_constructors_unchanged() -> None:
    # Backward-compat guard: legacy classes keep their original signatures.
    import inspect

    mrmr_params = list(inspect.signature(MRMR.__init__).parameters)
    assert mrmr_params == [
        "self",
        "n_features",
        "importance_function",
        "penalty_function",
        "random_state",
    ]
    # ModMRMR inherits MRMR's __init__ (no own constructor).
    assert "__init__" not in ModMRMR.__dict__
