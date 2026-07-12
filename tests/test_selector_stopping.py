"""Stopping-criterion tests for MRMRSelector (mechanism suite, Phase 1).

Fixed-count (n_features) vs score-threshold (data-driven, variable size). A fixed
relevance vector + a zero redundancy matrix make the per-step scores exactly equal
to the relevance values, so the stopping point is fully predictable.
"""

import numpy as np
import pandas as pd
import pytest

from modmrmr.core.estimator import MRMRSelector


def _frame(n_cols: int = 5, n: int = 200) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {f"f{i}": rng.normal(size=n) for i in range(n_cols)},
    )
    y = rng.normal(size=n)
    return X, y


def _fixed_relevance(values):
    def f(X, y, **_):  # noqa: ANN001
        return np.asarray(values, dtype=float)

    return f


def _zero_redundancy(X, **_):  # noqa: ANN001
    cols = list(X.columns)
    return pd.DataFrame(0.0, index=cols, columns=cols)


def _selector(values, **kwargs) -> MRMRSelector:
    return MRMRSelector(
        relevance=_fixed_relevance(values),
        redundancy=_zero_redundancy,
        operator="difference",
        task="regression",
        **kwargs,
    )


def test_score_threshold_stops_at_variable_size() -> None:
    # relevance [10,8,6,1,0.5]; threshold 5 keeps the first three, stops at 1.
    X, y = _frame()
    sel = _selector([10, 8, 6, 1, 0.5], n_features=None, score_threshold=5.0).fit(X, y)
    assert sel.n_selected_ == 3
    assert list(sel.selected_features_) == ["f0", "f1", "f2"]
    assert sel.selection_scores_ == pytest.approx([10.0, 8.0, 6.0])


def test_threshold_capped_by_n_features() -> None:
    # Even though 3 features clear the threshold, n_features=2 caps the size.
    X, y = _frame()
    sel = _selector([10, 8, 6, 1, 0.5], n_features=2, score_threshold=5.0).fit(X, y)
    assert sel.n_selected_ == 2
    assert list(sel.selected_features_) == ["f0", "f1"]


def test_threshold_selects_at_least_one_feature() -> None:
    # A threshold above every score still yields the single best feature (never empty).
    X, y = _frame()
    sel = _selector([10, 8, 6, 1, 0.5], n_features=None, score_threshold=100.0).fit(X, y)
    assert sel.n_selected_ == 1
    assert list(sel.selected_features_) == ["f0"]


def test_default_reproduces_fixed_k_and_sets_n_selected() -> None:
    # No threshold → today's fixed-k behavior; n_selected_ equals the count.
    X, y = _frame()
    sel = _selector([10, 8, 6, 1, 0.5], n_features=3).fit(X, y)
    assert sel.n_selected_ == 3
    assert len(sel.selected_features_) == 3
    assert list(sel.selected_features_) == ["f0", "f1", "f2"]


def test_n_features_none_without_threshold_raises() -> None:
    X, y = _frame()
    with pytest.raises(ValueError, match="n_features|score_threshold"):
        _selector([1, 2, 3, 4, 5], n_features=None, score_threshold=None).fit(X, y)


def test_n_selected_matches_selected_features() -> None:
    X, y = _frame()
    sel = _selector([10, 8, 6, 1, 0.5], n_features=None, score_threshold=2.0).fit(X, y)
    assert sel.n_selected_ == len(sel.selected_features_)
    assert sel.n_selected_ == len(sel.selected_idx_) == len(sel.selection_scores_)


def test_backward_compat_default_selector_unaffected() -> None:
    # The real engine (auto measures), default constructor, still selects n_features.
    X, y = _frame(n_cols=6)
    sel = MRMRSelector(n_features=4, random_state=0).fit(X, y)
    assert sel.n_selected_ == 4
    assert len(sel.selected_features_) == 4
