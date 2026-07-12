import pandas as pd

from mechanism.cost import runtime_by_family, runtime_vs_p


def test_runtime_by_family_groups_and_sorts():
    # Three families with deliberately out-of-order mean runtimes so the ascending
    # sort is actually exercised across the middle, not just first-vs-last.
    df = pd.DataFrame(
        {
            "relevance": [
                "mutual_info_sklearn",  # slowest family
                "mutual_info_sklearn",
                "pearson",  # fastest family
                "pearson",
                "distance_corr",  # middle family
                "distance_corr",
            ],
            "redundancy": ["pearson_abs"] * 6,
            "runtime_s": [4.0, 6.0, 0.1, 0.3, 1.0, 2.0],
            "dataset": ["a", "b", "a", "b", "a", "b"],
        }
    )
    out = runtime_by_family(df)
    assert list(out.columns) == ["family", "mean_runtime_s", "median_runtime_s", "n_fits"]
    # full monotonicity, not merely first <= last
    assert out["mean_runtime_s"].is_monotonic_increasing
    assert out["n_fits"].sum() == 6


def test_runtime_vs_p_orders_by_operator_then_p():
    # Three operators × three p-values, rows shuffled, so the (operator, p) two-level
    # sort is genuinely pinned — the operator level and the p level both matter.
    df = pd.DataFrame(
        {
            "p": [1000, 500, 2000, 2000, 500, 1000, 500, 2000, 1000],
            "operator": [
                "quotient",
                "multiplicative",
                "difference",
                "quotient",
                "difference",
                "difference",
                "quotient",
                "multiplicative",
                "multiplicative",
            ],
            "runtime_s": [2.0, 1.1, 3.0, 2.2, 3.1, 3.2, 1.0, 2.3, 2.1],
        }
    )
    out = runtime_vs_p(df)
    assert list(out.columns) == ["p", "operator", "mean_runtime_s", "n_fits"]
    # the output must be sorted by (operator, p) — check the actual tuple sequence,
    # which pins both the operator level and the within-operator p level
    tuples = list(zip(out["operator"], out["p"], strict=True))
    assert tuples == sorted(tuples)
    # operator is the primary key: all rows of one operator are contiguous and ordered
    assert out["operator"].is_monotonic_increasing
