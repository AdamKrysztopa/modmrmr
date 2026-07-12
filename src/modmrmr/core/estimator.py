"""General mRMR feature selectors as scikit-learn transformers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from inspect import signature
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.utils.validation import check_is_fitted

from modmrmr.core.combine import AGGREGATIONS, OPERATORS, aggregate, combine
from modmrmr.core.scorers import as_importance_function, as_penalty_matrix, get_scorer
from modmrmr.core.task import DEFAULT_MEASURES, detect_task

# Floor for the standard-MRMR mean-redundancy denominator to avoid divide-by-zero.
_MEAN_REDUNDANCY_FLOOR = 1e-3
# Lower clip for the Pearson redundancy matrix so no entry is exactly zero.
_CORR_CLIP_LOWER = 1e-5


def pearson_corr(X: pd.DataFrame) -> pd.DataFrame:
    """Absolute Pearson correlation matrix, clipped away from exact zero."""
    return X.corr().abs().clip(lower=_CORR_CLIP_LOWER)


class MRMR(BaseEstimator, TransformerMixin):
    """Standard mRMR: relevance / mean(redundancy vs already-selected)."""

    def __init__(
        self,
        n_features: int = 5,
        importance_function: Callable[..., Any] = mutual_info_regression,
        penalty_function: Callable[..., Any] = pearson_corr,
        random_state: int | None = None,
    ) -> None:
        self.n_features = n_features
        self.importance_function = importance_function
        self.penalty_function = penalty_function
        self.random_state = random_state

    def _get_scores(
        self, nom: pd.Series, denom: pd.DataFrame, selected: list[Any], not_selected: list[Any]
    ) -> pd.Series:
        if not selected:
            return nom.loc[not_selected]
        mean_red = denom.loc[not_selected, selected].mean(axis=1).fillna(_MEAN_REDUNDANCY_FLOOR)
        return nom.loc[not_selected] / mean_red

    def _resolve(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if self.random_state is not None and "random_state" in signature(func).parameters:
            return partial(func, random_state=self.random_state)
        return func

    @staticmethod
    def _to_frame(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        raise TypeError("X must be a numpy ndarray or pandas DataFrame.")

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        **fit_params: Any,
    ) -> MRMR:
        X_df = self._to_frame(X)
        n_select = min(self.n_features, X_df.shape[1])
        imp_func = self._resolve(self.importance_function)
        pen_func = self._resolve(self.penalty_function)

        nom = pd.Series(imp_func(X_df, y, **fit_params), index=X_df.columns)
        denom = pen_func(X_df)

        selected: list[Any] = []
        not_selected: list[Any] = list(X_df.columns)
        for _ in range(n_select):
            score = self._get_scores(nom, denom, selected, not_selected)
            best = score.idxmax()
            selected.append(best)
            not_selected.remove(best)

        self.selected_idx_ = [X_df.columns.get_loc(name) for name in selected]
        self.n_features_in_ = X_df.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns)
            self.selected_features_ = np.asarray(X.columns)[self.selected_idx_]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        check_is_fitted(self, attributes=["selected_idx_"])
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_idx_]
        if isinstance(X, np.ndarray):
            return X[:, self.selected_idx_]
        raise TypeError("X must be a numpy ndarray or pandas DataFrame.")


class ModMRMR(MRMR):
    """Modified mRMR: relevance * (1 - max(redundancy vs already-selected)).

    The ``(1 - max_redundancy)`` factor is clipped at 0 so unbounded penalty
    functions cannot drive the score negative.
    """

    def _get_scores(
        self, nom: pd.Series, denom: pd.DataFrame, selected: list[Any], not_selected: list[Any]
    ) -> pd.Series:
        if not selected:
            return nom.loc[not_selected]
        max_red = denom.loc[not_selected, selected].max(axis=1).fillna(0.0)
        return nom.loc[not_selected] * (1.0 - max_red).clip(lower=0.0)


# Relevance measures usable by NAME straight from sklearn (Plan B later promotes
# these to registered scorers; Plan A does not block on that). f_* return a
# (F, p_values) tuple, so only element [0] is the per-feature relevance.
_SKLEARN_RELEVANCE: dict[str, Callable[..., Any]] = {
    "f_classif": lambda X, y, **_: f_classif(X, y)[0],
    "f_regression": lambda X, y, **_: f_regression(X, y)[0],
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
}


def _resolve_relevance(
    relevance: Any, task: str, random_state: int | None = None
) -> Callable[..., Any]:
    """Return an ``importance_function(X, y, **) -> per-feature array``.

    ``relevance`` may be ``"auto"`` (use ``DEFAULT_MEASURES[task]``), a callable
    (used as-is), a known sklearn relevance name, or a registered scorer name.
    For the registered-name path, ``random_state`` is threaded into the scorer
    adapter (its closure swallows ``random_state`` in ``**_``, so the selector's
    seed cannot be injected after the fact — it must be bound here).
    """
    if relevance == "auto":
        relevance = DEFAULT_MEASURES[task][0]
    if callable(relevance):
        return relevance
    if relevance in _SKLEARN_RELEVANCE:
        return _SKLEARN_RELEVANCE[relevance]
    return as_importance_function(get_scorer(relevance), **_seed_kwarg(random_state))


def _resolve_redundancy(
    redundancy: Any, task: str, random_state: int | None = None
) -> Callable[..., Any]:
    """Return a ``penalty_function(X, **) -> feature x feature DataFrame``.

    ``redundancy`` may be ``"auto"`` (use ``DEFAULT_MEASURES[task]``), a callable
    (used as-is), or a registered scorer name. For the registered-name path,
    ``random_state`` is threaded into the scorer adapter (see
    :func:`_resolve_relevance`).
    """
    if redundancy == "auto":
        redundancy = DEFAULT_MEASURES[task][1]
    if callable(redundancy):
        return redundancy
    return as_penalty_matrix(get_scorer(redundancy), **_seed_kwarg(random_state))


def _seed_kwarg(random_state: int | None) -> dict[str, int]:
    """``{"random_state": rs}`` when a seed is set, else ``{}`` (keeps the adapter default)."""
    return {} if random_state is None else {"random_state": random_state}


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Configurable greedy mRMR selector spanning the full 4-axis design space.

    The two legacy classes are thin presets of this engine:
    ``MRMR`` == ``operator="quotient", aggregation="mean"`` and
    ``ModMRMR`` == ``operator="multiplicative", aggregation="max"`` (with the
    legacy relevance/redundancy measures). See ``tests/test_selector_parity.py``.

    Args:
        n_features: fixed number of features to select (clamped to the input
            width). May be ``None`` to select a data-driven, variable-size set
            governed entirely by ``score_threshold``; when both are given,
            ``n_features`` is an upper cap.
        relevance: ``"auto"``, a callable ``(X, y, **) -> array``, a known
            sklearn name (``"f_classif"``, ``"f_regression"``,
            ``"mutual_info_classif"``, ``"mutual_info_regression"``), or a
            registered scorer name.
        redundancy: ``"auto"``, a callable ``(X, **) -> DataFrame``, or a
            registered scorer name.
        operator: ``"difference"`` | ``"quotient"`` | ``"multiplicative"``.
        aggregation: ``"mean"`` | ``"max"`` | ``"sum"``.
        task: ``"auto"`` (infer via :func:`detect_task`) | ``"classification"``
            | ``"regression"``.
        random_state: passed to any relevance/redundancy callable that accepts a
            ``random_state`` parameter (matches legacy MRMR behaviour).
        score_threshold: if set, greedy selection stops before adding a feature
            whose best per-step combined score falls below this value (still
            capped by ``n_features`` when that is also given). The first feature
            is always selected, so a fitted selector is never empty. ``None``
            (default) reproduces fixed-count selection exactly. Setting both
            ``n_features=None`` and ``score_threshold=None`` raises ``ValueError``.

    Fitted attribute ``n_selected_`` records the chosen number of features
    (equal to ``n_features`` in fixed-count mode, data-driven under a threshold).
    """

    def __init__(
        self,
        n_features: int | None = 5,
        relevance: Any = "auto",
        redundancy: Any = "auto",
        operator: str = "difference",
        aggregation: str = "mean",
        task: str = "auto",
        random_state: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.n_features = n_features
        self.relevance = relevance
        self.redundancy = redundancy
        self.operator = operator
        self.aggregation = aggregation
        self.task = task
        self.random_state = random_state
        self.score_threshold = score_threshold

    @staticmethod
    def _to_frame(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        raise TypeError("X must be a numpy ndarray or pandas DataFrame.")

    def _resolve_random_state(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if self.random_state is not None and "random_state" in signature(func).parameters:
            return partial(func, random_state=self.random_state)
        return func

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        **fit_params: Any,
    ) -> MRMRSelector:
        if self.n_features is None and self.score_threshold is None:
            raise ValueError(
                "Set n_features (fixed count) or score_threshold (data-driven size); both are None."
            )
        if self.operator not in OPERATORS:
            raise ValueError(
                f"Unknown operator {self.operator!r}. Expected one of: {', '.join(OPERATORS)}."
            )
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {self.aggregation!r}. "
                f"Expected one of: {', '.join(AGGREGATIONS)}."
            )
        X_df = self._to_frame(X)
        self.task_ = detect_task(y) if self.task == "auto" else self.task

        imp_func = self._resolve_random_state(
            _resolve_relevance(self.relevance, self.task_, self.random_state)
        )
        pen_func = self._resolve_random_state(
            _resolve_redundancy(self.redundancy, self.task_, self.random_state)
        )

        relevance = pd.Series(imp_func(X_df, y, **fit_params), index=X_df.columns)
        redundancy = pen_func(X_df)

        # Cap on the number of features: all of them when n_features is None (a
        # purely threshold-driven, variable-size selection), else the fixed count.
        cap = X_df.shape[1] if self.n_features is None else min(self.n_features, X_df.shape[1])

        selected: list[Any] = []
        not_selected: list[Any] = list(X_df.columns)
        scores: list[float] = []
        for step_idx in range(cap):
            if not selected:
                step = relevance.loc[not_selected]
            else:
                block = redundancy.loc[not_selected, selected]
                step = combine(
                    relevance.loc[not_selected],
                    aggregate(block, self.aggregation),
                    self.operator,
                )
            best = step.idxmax()
            best_score = float(step.loc[best])
            # Threshold stopping: stop before adding a feature whose best combined
            # score is below the threshold. The first feature is always selected
            # (step_idx == 0), so a fitted selector never returns an empty set.
            if (
                self.score_threshold is not None
                and step_idx > 0
                and best_score < self.score_threshold
            ):
                break
            scores.append(best_score)
            selected.append(best)
            not_selected.remove(best)

        self.selected_idx_ = [X_df.columns.get_loc(name) for name in selected]
        self.selection_order_ = list(selected)
        self.selection_scores_ = scores
        self.n_selected_ = len(selected)
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = np.asarray(X_df.columns)
        self.selected_features_ = np.asarray(X_df.columns)[self.selected_idx_]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        check_is_fitted(self, attributes=["selected_idx_"])
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_idx_]
        if isinstance(X, np.ndarray):
            return X[:, self.selected_idx_]
        raise TypeError("X must be a numpy ndarray or pandas DataFrame.")
