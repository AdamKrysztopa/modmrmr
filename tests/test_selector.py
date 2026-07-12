"""Tests for the configurable MRMRSelector engine (Plan A, Tasks 3-4)."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from modmrmr.core.estimator import MRMRSelector


def _regression_frame(n: int = 400):
    rng = np.random.default_rng(0)
    signal_a = rng.normal(size=n)
    signal_b = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "relevant_a": signal_a + rng.normal(scale=0.1, size=n),
            "duplicate_a": signal_a + rng.normal(scale=0.1, size=n),
            "relevant_b": signal_b + rng.normal(scale=0.1, size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = signal_a + signal_b
    return X, y


def test_regression_auto_task_and_default_measures() -> None:
    X, y = _regression_frame()
    sel = MRMRSelector(n_features=2, random_state=0).fit(X, y)
    assert sel.task_ == "regression"
    # relevant_b (independent signal) must survive; the A near-duplicates
    # must not both be picked.
    picks = set(sel.selected_features_)
    assert "relevant_b" in picks
    assert not {"relevant_a", "duplicate_a"}.issubset(picks)


def test_fitted_attributes_present_and_consistent() -> None:
    X, y = _regression_frame()
    sel = MRMRSelector(n_features=3, random_state=0).fit(X, y)
    assert sel.n_features_in_ == 4
    assert list(sel.feature_names_in_) == list(X.columns)
    assert len(sel.selected_idx_) == 3
    assert len(sel.selection_order_) == 3
    assert len(sel.selection_scores_) == 3
    # selected_features_ must equal columns indexed by selected_idx_, in order.
    assert list(sel.selected_features_) == [X.columns[i] for i in sel.selected_idx_]
    assert list(sel.selected_features_) == list(sel.selection_order_)
    # Every selection score is a finite float.
    assert all(isinstance(s, float) and np.isfinite(s) for s in sel.selection_scores_)


def test_numpy_input_exposes_integer_feature_names() -> None:
    X, y = _regression_frame()
    sel = MRMRSelector(n_features=2, random_state=0).fit(X.to_numpy(), np.asarray(y))
    assert sel.n_features_in_ == 4
    assert list(sel.feature_names_in_) == [0, 1, 2, 3]
    assert sel.transform(X.to_numpy()).shape == (len(y), 2)


def test_transform_preserves_dataframe_columns() -> None:
    X, y = _regression_frame()
    sel = MRMRSelector(n_features=2, random_state=0).fit(X, y)
    out = sel.transform(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(sel.selected_features_)


def test_relevance_and_redundancy_accept_registered_names() -> None:
    X, y = _regression_frame()
    sel = MRMRSelector(
        n_features=2,
        relevance="mutual_info_sklearn",
        redundancy="pearson_abs",
        operator="quotient",
        aggregation="mean",
        task="regression",
        random_state=0,
    ).fit(X, y)
    assert len(sel.selected_features_) == 2


def test_relevance_and_redundancy_accept_callables() -> None:
    from sklearn.feature_selection import mutual_info_regression

    from modmrmr.core.estimator import pearson_corr

    X, y = _regression_frame()
    sel = MRMRSelector(
        n_features=2,
        relevance=mutual_info_regression,
        redundancy=pearson_corr,
        operator="quotient",
        aggregation="mean",
        task="regression",
        random_state=0,
    ).fit(X, y)
    assert len(sel.selected_features_) == 2


def test_n_features_clamped_to_width() -> None:
    X, y = _regression_frame()
    sel = MRMRSelector(n_features=99, random_state=0).fit(X, y)
    assert len(sel.selected_features_) == 4  # only 4 columns exist


def test_classification_auto_task_selects_informative_features() -> None:
    X_arr, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        random_state=0,
    )
    cols = [f"f{i}" for i in range(X_arr.shape[1])]
    X = pd.DataFrame(X_arr, columns=cols)
    sel = MRMRSelector(n_features=3, random_state=0).fit(X, y)
    assert sel.task_ == "classification"
    assert len(sel.selected_features_) == 3
    # All picks come from the real column set (no crash on the f_classif path).
    assert set(sel.selected_features_).issubset(set(cols))


def test_classification_with_string_labels() -> None:
    X_arr, y_int = make_classification(n_samples=200, n_features=6, n_informative=3, random_state=1)
    y = np.array(["low", "mid", "high"])[y_int % 3]
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(6)])
    sel = MRMRSelector(n_features=2, random_state=0).fit(X, y)
    assert sel.task_ == "classification"
    assert len(sel.selected_features_) == 2


def test_explicit_classification_task_override() -> None:
    # Integer target with MANY uniques would auto-detect as regression; the
    # explicit task="classification" override forces the classification measures.
    X_arr, y = make_classification(n_samples=200, n_features=6, n_informative=3, random_state=2)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(6)])
    sel = MRMRSelector(
        n_features=2, relevance="mutual_info_classif", task="classification", random_state=0
    ).fit(X, y)
    assert sel.task_ == "classification"
    assert len(sel.selected_features_) == 2


def test_random_state_is_threaded_into_registered_scorer() -> None:
    """Regression: the selector seed must reach a registered scorer used by NAME.

    The scorer adapter closure swallows ``random_state`` in ``**_``, so the seed
    cannot be injected via signature-sniffing after construction — it must be
    bound when the adapter is built. A recording stub confirms the actual seed
    (not the adapter's hardcoded default of 42) is what the scorer receives.
    """
    from modmrmr.core.scorers import _RawScore, list_scorers, register_scorer

    seen: list[int | None] = []

    class _RecordingScorer:
        name = "recording_seed_probe"

        def score_pair(self, x, y, *, random_state=None) -> _RawScore:
            seen.append(random_state)
            return _RawScore(raw_value=0.0, n_pairs=len(x))

    register_scorer("recording_seed_probe", _RecordingScorer())
    try:
        X, y = _regression_frame()
        MRMRSelector(
            n_features=1,
            relevance="recording_seed_probe",
            redundancy="pearson_abs",
            task="regression",
            random_state=123,
        ).fit(X, y)
        assert seen, "the recording scorer was never called"
        assert set(seen) == {123}, f"expected seed 123 to be threaded, saw {set(seen)}"
    finally:
        # Keep the global registry clean for other tests.
        assert "recording_seed_probe" in list_scorers()
        from modmrmr.core.scorers import base as _scorer_base

        _scorer_base._SCORER_REGISTRY.pop("recording_seed_probe", None)
