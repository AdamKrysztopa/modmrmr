import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scikit_posthocs")

from mechanism.stats import FriedmanNemenyiResult, friedman_nemenyi, plot_critical_difference


def _long_df(seed: int = 0, dominant: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for block in range(12):
        base = rng.uniform(0.3, 0.6)
        for t in ["a", "b", "c", "d"]:
            lift = 0.3 if (dominant and t == "a") else 0.0
            rows.append(
                {"method": t, "dataset": f"d{block}", "f1": base + lift + rng.normal(0, 0.01)}
            )
    return pd.DataFrame(rows)


def test_dominant_treatment_gets_best_rank_and_significance():
    res = friedman_nemenyi(_long_df(), treatment="method", block="dataset", value="f1")
    assert isinstance(res, FriedmanNemenyiResult)
    assert res.avg_ranks.idxmin() == "a"
    assert res.p_value < 0.01
    assert res.n_blocks == 12
    assert res.nemenyi_p.loc["a", "b"] < 0.05


def test_indistinguishable_treatments_not_significant():
    res = friedman_nemenyi(
        _long_df(seed=1, dominant=False), treatment="method", block="dataset", value="f1"
    )
    assert res.p_value > 0.05


def test_lower_is_better_flips_ranks():
    df = _long_df()
    hi = friedman_nemenyi(df, treatment="method", block="dataset", value="f1")
    lo = friedman_nemenyi(
        df, treatment="method", block="dataset", value="f1", higher_is_better=False
    )
    assert hi.avg_ranks.idxmin() == "a"
    assert lo.avg_ranks.idxmax() == "a"


def test_cd_diagram_writes_file(tmp_path):
    res = friedman_nemenyi(_long_df(), treatment="method", block="dataset", value="f1")
    out = plot_critical_difference(res, tmp_path / "cd.pdf", title="test")
    assert out.exists() and out.stat().st_size > 0
