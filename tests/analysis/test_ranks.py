import numpy as np
import pandas as pd

from analysis.ranks import (
    auc_k_table,
    mean_ranks,
    method_rank_table,
    primary_metric,
)


def test_primary_metric() -> None:
    assert primary_metric("classification") == "balanced_accuracy"
    assert primary_metric("regression") == "r2"


def test_auc_k_table_shape_and_orientation(synthetic_results: pd.DataFrame) -> None:
    tbl = auc_k_table(synthetic_results, "classification")
    assert tbl.index.name == "dataset"
    assert "ModMRMR" in tbl.columns
    assert not tbl.isna().any().any()


def test_mean_ranks_best_score_gets_rank_one() -> None:
    # method "A" strictly dominates on every dataset -> mean rank 1.0
    mat = pd.DataFrame(
        {"A": [0.9, 0.8, 0.95], "B": [0.5, 0.4, 0.6], "C": [0.1, 0.2, 0.3]},
        index=["d1", "d2", "d3"],
    )
    ranks = mean_ranks(mat)
    assert ranks.index[0] == "A"
    assert np.isclose(ranks.loc["A"], 1.0)
    assert np.isclose(ranks.loc["C"], 3.0)


def test_modmrmr_ranks_well_on_synthetic(synthetic_results: pd.DataFrame) -> None:
    tbl = auc_k_table(synthetic_results, "classification")
    ranks = mean_ranks(tbl)
    # synthetic ModMRMR has the highest base_skill -> best mean rank
    assert ranks.index[0] == "ModMRMR"


def test_method_rank_table_has_mean_row(synthetic_results: pd.DataFrame) -> None:
    rt = method_rank_table(synthetic_results, "regression")
    assert "mean_rank" in rt.index
