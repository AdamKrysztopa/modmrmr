import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1

from benchmarks.cells import SelectKBestAdapter
from benchmarks.protocol import (
    RESULT_COLUMNS,
    _detect_task,
    default_learners,
    k_curve,
    run_grid,
    stability,
    stability_index,
)

# The canonical 17-column results schema.
# run_grid MUST emit exactly this, in this order.
CANONICAL_RESULT_COLUMNS = [
    "dataset",
    "task",
    "method",
    "operator",
    "aggregation",
    "relevance",
    "redundancy",
    "n_samples",
    "n_features",
    "k",
    "learner",
    "seed",
    "fold",
    "metric",
    "score",
    "stability",
    "runtime_s",
]


def test_stability_index_perfect_agreement_is_one():
    # Every resample picks the same 3 of 10 features.
    Z = np.zeros((8, 10))
    Z[:, [0, 1, 2]] = 1.0
    assert stability_index(Z) == 1.0


def test_stability_index_is_bounded_above_by_one():
    rng = np.random.default_rng(0)
    Z = (rng.random((20, 30)) < 0.2).astype(float)
    val = stability_index(Z)
    assert val <= 1.0 + 1e-9
    assert val > -1.0  # random selection sits near 0, never wildly negative here


def test_stability_index_random_near_zero():
    rng = np.random.default_rng(1)
    # k=5 of 50 chosen uniformly at random each of 40 draws -> ~0.
    Z = np.zeros((40, 50))
    for i in range(40):
        Z[i, rng.choice(50, size=5, replace=False)] = 1.0
    assert abs(stability_index(Z)) < 0.1


def test_detect_task_heuristic():
    assert _detect_task(pd.Series(["a", "b", "a"])) == "classification"
    assert _detect_task(pd.Series([0, 1, 0, 1, 0])) == "classification"
    assert _detect_task(pd.Series(np.linspace(0.0, 1.0, 200))) == "regression"


def test_stability_of_selectkbest_is_in_range():
    X, y = make_classification(n_samples=200, n_features=20, n_informative=6, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    adapter = SelectKBestAdapter("skb_f", "f")
    val = stability(adapter, X, pd.Series(y), n_boot=6, k=5)
    assert -1.0 <= val <= 1.0


def test_stability_is_deterministic():
    X, y = make_classification(n_samples=150, n_features=20, n_informative=6, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    adapter = SelectKBestAdapter("skb_f", "f")
    a = stability(adapter, X, pd.Series(y), n_boot=6, k=5)
    b = stability(adapter, X, pd.Series(y), n_boot=6, k=5)
    assert a == b


def test_default_learners_have_three_estimators_each():
    for task in ("classification", "regression"):
        learners = default_learners(task)
        assert set(learners) == {"knn", "linear", "rf"}


def test_k_curve_returns_long_format_columns():
    X, y = make_classification(n_samples=200, n_features=20, n_informative=6, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    adapter = SelectKBestAdapter("skb_f", "f")
    df = k_curve(adapter, X, pd.Series(y), "classification", ks=[2, 5], learners=None, cv=3)
    assert set(df.columns) == {"k", "learner", "fold", "metric", "score"}
    assert set(df["k"]) == {2, 5}
    assert set(df["learner"]) == {"knn", "linear", "rf"}
    # 2 ks * 3 learners * 3 folds = 18 rows
    assert len(df) == 18
    assert df["metric"].unique().tolist() == ["balanced_accuracy"]


def test_k_curve_is_monotone_ish_on_friedman1():
    # On Friedman1, 5 informative features; going from k=1 to k=5 must help R^2.
    X, y = make_friedman1(n_samples=300, n_features=10, noise=0.1, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    adapter = SelectKBestAdapter("skb_f", "f")
    df = k_curve(adapter, X, pd.Series(y), "regression", ks=[1, 5], learners=None, cv=3)
    mean_by_k = df.groupby("k")["score"].mean()
    assert mean_by_k.loc[5] > mean_by_k.loc[1]
    assert df["metric"].unique().tolist() == ["r2"]


def test_run_grid_columns_equal_canonical_schema_in_exact_order(monkeypatch):
    import benchmarks.protocol as protocol_mod

    def fake_load(name):
        Xa, ya = make_classification(n_samples=120, n_features=12, n_informative=5, random_state=0)
        return (
            pd.DataFrame(Xa, columns=[f"f{i}" for i in range(12)]),
            pd.Series(ya),
            "classification",
        )

    monkeypatch.setattr(protocol_mod, "load_dataset", fake_load)
    adapter = SelectKBestAdapter("skb_f", "f")
    df = run_grid(cells=[adapter], datasets=["fake_a"], learners=None, cv=3, seeds=[0])
    # Contract conformance: exact columns, exact order.
    assert list(df.columns) == CANONICAL_RESULT_COLUMNS
    assert RESULT_COLUMNS == CANONICAL_RESULT_COLUMNS


def test_run_grid_produces_tidy_long_dataframe(monkeypatch):
    import benchmarks.protocol as protocol_mod

    # Avoid any download: stub load_dataset with a tiny synthetic classification set.
    def fake_load(name):
        Xa, ya = make_classification(n_samples=120, n_features=12, n_informative=5, random_state=0)
        return (
            pd.DataFrame(Xa, columns=[f"f{i}" for i in range(12)]),
            pd.Series(ya),
            "classification",
        )

    monkeypatch.setattr(protocol_mod, "load_dataset", fake_load)

    adapter = SelectKBestAdapter("skb_f", "f")
    df = run_grid(
        cells=[adapter],
        datasets=["fake_a", "fake_b"],
        learners=None,
        cv=3,
        seeds=[0, 1],
    )
    assert set(df.columns) == set(CANONICAL_RESULT_COLUMNS)
    assert set(df["dataset"]) == {"fake_a", "fake_b"}
    assert set(df["method"]) == {"skb_f"}
    assert set(df["seed"]) == {0, 1}
    assert df["task"].unique().tolist() == ["classification"]
    # Baselines leave the MRMR-family metadata blank.
    assert set(df["operator"]) == {""}
    assert set(df["redundancy"]) == {""}
    # Stability is repeated per fold row and finite in [-1, 1].
    assert df["stability"].between(-1.0, 1.0).all()
    assert (df["runtime_s"] >= 0.0).all()
    assert len(df) > 0
