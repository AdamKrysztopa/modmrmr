import pandas as pd

from analysis.schema import (
    PRIMARY_METRIC,
    RESULT_COLUMNS,
    data_regime,
    make_synthetic_results,
)
from benchmarks.protocol import RESULT_COLUMNS as BENCHMARKS_RESULT_COLUMNS


def test_primary_metric_map() -> None:
    assert PRIMARY_METRIC["classification"] == "balanced_accuracy"
    assert PRIMARY_METRIC["regression"] == "r2"


def test_data_regime() -> None:
    assert data_regime(n_samples=100, n_features=5000) == "p>>n"
    assert data_regime(n_samples=5000, n_features=8) == "n>=p"


def test_result_columns_matches_benchmarks_protocol() -> None:
    # Plan D's schema MUST NOT drift from Plan C's canonical results schema —
    # both are pinned to the interface contract's CANONICAL RESULTS SCHEMA.
    assert RESULT_COLUMNS == BENCHMARKS_RESULT_COLUMNS


def test_synthetic_results_matches_schema() -> None:
    df = make_synthetic_results()
    assert list(df.columns) == RESULT_COLUMNS
    assert set(df["task"]) == {"classification", "regression"}
    assert df["method"].nunique() >= 4
    # primary metric present for both tasks
    for task, metric in PRIMARY_METRIC.items():
        assert not df[(df["task"] == task) & (df["metric"] == metric)].empty
    assert not df.isna().any().any()
    assert isinstance(df, pd.DataFrame)
