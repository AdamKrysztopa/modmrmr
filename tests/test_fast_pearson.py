import numpy as np
import pandas as pd
import pytest

from modmrmr.core.estimator import _resolve_redundancy
from modmrmr.core.scorers.base import fast_pearson_penalty


def _frame(seed: int = 0, n: int = 120, p: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    df["const"] = 1.0  # constant column: reference loop scores it 0 off-diagonal
    return df


def test_matches_reference_pair_loop():
    X = _frame()
    ref = _resolve_redundancy("pearson_abs", "regression")(X)
    fast = fast_pearson_penalty(X)
    pd.testing.assert_frame_equal(fast, ref, check_exact=False, atol=1e-10)


def test_diagonal_is_one_and_symmetric():
    X = _frame(seed=1)
    m = fast_pearson_penalty(X)
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
    assert np.allclose(m.to_numpy(), m.to_numpy().T)


def test_rejects_non_finite_input():
    X = _frame(seed=2)
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fast_pearson_penalty(X)


@pytest.mark.perf_budget
def test_p2000_under_five_seconds():
    from time import perf_counter

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 2000)))
    t0 = perf_counter()
    fast_pearson_penalty(X)
    assert perf_counter() - t0 < 5.0
