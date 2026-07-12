import pandas as pd

from analysis.guidance import build_decision_guide, rank_criteria
from analysis.schema import RESULT_COLUMNS


def _row(**kw: object) -> dict[str, object]:
    base = dict.fromkeys(RESULT_COLUMNS, "")
    base.update(
        {
            "n_samples": 0,
            "n_features": 0,
            "k": 5,
            "seed": 0,
            "score": 0.0,
            "stability": 0.5,
            "runtime_s": 1.0,
        }
    )
    base.update(kw)
    return base


def test_rank_criteria_orders_by_mean_auc(synthetic_results: pd.DataFrame) -> None:
    ranks = rank_criteria(synthetic_results, "classification")
    assert ranks.index[0] == "ModMRMR"  # highest synthetic skill
    assert ranks.is_monotonic_increasing


def test_build_decision_guide_keys_and_recommendation() -> None:
    # Two classification datasets, one per regime; method WIN dominates.
    rows: list[dict[str, object]] = []
    specs = [
        ("small", 500, 30, "n>=p"),  # p<=n
        ("wide", 50, 5000, "p>>n"),  # p>>n
    ]
    for dataset, n, p, _regime in specs:
        for method, score in [("WIN", 0.9), ("LOSE", 0.4)]:
            rows.append(
                _row(
                    dataset=dataset,
                    task="classification",
                    method=method,
                    n_samples=n,
                    n_features=p,
                    metric="balanced_accuracy",
                    score=score,
                )
            )
    guide = build_decision_guide(pd.DataFrame(rows, columns=RESULT_COLUMNS))
    assert ("classification", "n>=p") in guide
    assert ("classification", "p>>n") in guide
    for entry in guide.values():
        assert entry["recommended"] == "WIN"
        assert entry["ranking"][0][0] == "WIN"
