"""MRMRSelector is public and drops into an sklearn Pipeline (Plan A, Task 6)."""

import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


def test_mrmrselector_exported_at_top_level() -> None:
    import modmrmr

    assert "MRMRSelector" in modmrmr.__all__
    from modmrmr import MRMRSelector  # noqa: F401


def test_pipeline_classification_end_to_end() -> None:
    from modmrmr import MRMRSelector

    X_arr, y = make_classification(
        n_samples=300, n_features=10, n_informative=4, n_redundant=3, random_state=0
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])
    pipe = Pipeline(
        [
            ("select", MRMRSelector(n_features=4, random_state=0)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    ).fit(X, y)
    assert pipe.predict(X).shape == (300,)
    assert pipe.named_steps["select"].task_ == "classification"


def test_pipeline_regression_end_to_end() -> None:
    from modmrmr import MRMRSelector

    X_arr, y = make_regression(
        n_samples=300, n_features=10, n_informative=4, noise=0.1, random_state=0
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])
    pipe = Pipeline(
        [
            ("select", MRMRSelector(n_features=4, random_state=0)),
            ("lr", LinearRegression()),
        ]
    ).fit(X, y)
    assert pipe.predict(X).shape == (300,)
    assert pipe.named_steps["select"].task_ == "regression"


def test_clone_preserves_params() -> None:
    from modmrmr import MRMRSelector

    original = MRMRSelector(
        n_features=3, operator="multiplicative", aggregation="max", random_state=7
    )
    cloned = clone(original)
    assert cloned.get_params() == original.get_params()
