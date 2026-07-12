import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from benchmarks.stats import average_ranks, critical_difference, friedman_pvalue  # noqa: E402


def _synthetic_ranked_results(n_datasets=10):
    # A dominates B dominates C on every dataset -> Friedman should be significant.
    rows = []
    rng = np.random.default_rng(0)
    for d in range(n_datasets):
        base = rng.uniform(0.5, 0.6)
        for method, bump in [("A", 0.20), ("B", 0.10), ("C", 0.00)]:
            rows.append(
                {
                    "dataset": f"ds{d}",
                    "method": method,
                    "task": "classification",
                    "score": base + bump,
                }
            )
    return pd.DataFrame(rows)


def test_friedman_pvalue_is_significant_when_one_method_dominates():
    results = _synthetic_ranked_results()
    p = friedman_pvalue(results, "classification")
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_average_ranks_order_matches_dominance():
    results = _synthetic_ranked_results()
    ranks = average_ranks(results, "classification")
    assert ranks["A"] < ranks["B"] < ranks["C"]
    assert abs(ranks["A"] - 1.0) < 1e-9  # A best on every dataset -> rank 1


def test_critical_difference_returns_a_figure():
    results = _synthetic_ranked_results()
    fig = critical_difference(results, "classification")
    assert isinstance(fig, Figure)
