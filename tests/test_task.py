"""Unit tests for task detection + default measure map (Plan A, Task 2)."""

import numpy as np
import pandas as pd

from modmrmr.core.task import DEFAULT_MEASURES, detect_task


def test_continuous_float_is_regression() -> None:
    y = np.linspace(0.0, 10.0, 200)  # 200 unique continuous values
    assert detect_task(y) == "regression"


def test_few_unique_integer_is_classification() -> None:
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0] * 25)  # 3 classes, integer dtype
    assert detect_task(y) == "classification"


def test_many_unique_integer_is_regression() -> None:
    # integer dtype but n_unique (500) far above max(20, 0.05*n=25) -> regression
    y = np.arange(500)
    assert detect_task(y) == "regression"


def test_string_labels_are_classification() -> None:
    y = np.array(["cat", "dog", "cat", "dog", "fish"])
    assert detect_task(y) == "classification"


def test_pandas_categorical_is_classification() -> None:
    y = pd.Series(["a", "b", "a", "c"], dtype="category")
    assert detect_task(y) == "classification"


def test_boolean_is_classification() -> None:
    y = np.array([True, False, True, True, False])
    assert detect_task(y) == "classification"


def test_default_measures_map() -> None:
    assert DEFAULT_MEASURES == {
        "classification": ("f_classif", "pearson_abs"),
        "regression": ("f_regression", "pearson_abs"),
    }
