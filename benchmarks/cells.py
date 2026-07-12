"""Feature-selection cells and baseline adapters.

Every runnable method is a *SelectorAdapter*: an object with a ``name`` and a
``select(X, y, task, k) -> list[int]`` method returning the indices (into ``X``'s
columns) of the ``k`` chosen features. This tiny interface lets MRMR-family cells and
external baselines share one CV loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


@runtime_checkable
class SelectorAdapter(Protocol):
    """A uniform feature-selection interface for the benchmark grid."""

    name: str

    def select(self, X: pd.DataFrame, y: pd.Series, task: str, k: int) -> list[int]:
        """Return indices of the k selected columns of X (train fold only)."""
        ...


_SKB_SCORE = {
    ("f", "classification"): f_classif,
    ("f", "regression"): f_regression,
    ("mutual_info", "classification"): mutual_info_classif,
    ("mutual_info", "regression"): mutual_info_regression,
}


@dataclass
class SelectKBestAdapter:
    """Univariate relevance-only control (no redundancy term)."""

    name: str
    score: str  # "f" | "mutual_info"

    def select(self, X: pd.DataFrame, y: pd.Series, task: str, k: int) -> list[int]:
        score_fn = _SKB_SCORE[(self.score, task)]
        # mutual_info_* draw from the global RNG unless seeded — pin the seed so the
        # baseline (and the stability bootstrap that refits it) is reproducible.
        if self.score == "mutual_info":
            score_fn = partial(score_fn, random_state=0)
        k_eff = min(k, X.shape[1])
        skb = SelectKBest(score_func=score_fn, k=k_eff).fit(X.to_numpy(), y.to_numpy())
        idx = np.flatnonzero(skb.get_support())
        return [int(i) for i in idx]


# --------------------------------------------------------------------------- #
# MRMR-family cells (consume Plan A MRMRSelector + Plan B scorers)
# --------------------------------------------------------------------------- #
_RELEVANCE_BY_FAMILY = {
    ("mi", "classification"): "mutual_info_classif",
    ("mi", "regression"): "mutual_info_sklearn",
    ("f", "classification"): "f_classif",
    ("f", "regression"): "f_regression",
}


@dataclass
class MRMRCellAdapter:
    """A single MRMR design-space cell, run through Plan A's MRMRSelector."""

    name: str
    relevance_family: str  # "mi" | "f"
    redundancy: str  # registered scorer name, e.g. "pearson_abs" | "distance_corr"
    operator: str  # "difference" | "quotient" | "multiplicative"
    aggregation: str  # "mean" | "max" | "sum"

    def _resolve_relevance(self, task: str) -> str:
        return _RELEVANCE_BY_FAMILY[(self.relevance_family, task)]

    def select(self, X: pd.DataFrame, y: pd.Series, task: str, k: int) -> list[int]:
        from modmrmr.core.estimator import MRMRSelector  # local import: Plan A dep

        selector = MRMRSelector(
            n_features=min(k, X.shape[1]),
            relevance=self._resolve_relevance(task),
            redundancy=self.redundancy,
            operator=self.operator,
            aggregation=self.aggregation,
            task=task,
            random_state=0,
        )
        selector.fit(X, y)
        return [int(i) for i in selector.selected_idx_]


# --------------------------------------------------------------------------- #
# External baseline adapters
# --------------------------------------------------------------------------- #
@dataclass
class MRMRSelectionBaseline:
    """smazzanti/mrmr reference MRMR (F-stat relevance, correlation redundancy)."""

    name: str

    def select(self, X: pd.DataFrame, y: pd.Series, task: str, k: int) -> list[int]:
        from mrmr import mrmr_classif, mrmr_regression  # local import

        k_eff = min(k, X.shape[1])
        fn = mrmr_classif if task == "classification" else mrmr_regression
        selected = fn(X=X, y=y, K=k_eff, show_progress=False)
        col_to_idx = {c: i for i, c in enumerate(X.columns)}
        return [int(col_to_idx[c]) for c in selected]


@dataclass
class ReliefFBaseline:
    """skrebate ReliefF (instance-based relevance, no redundancy term)."""

    name: str

    def select(self, X: pd.DataFrame, y: pd.Series, task: str, k: int) -> list[int]:
        from skrebate import ReliefF  # local import

        k_eff = min(k, X.shape[1])
        n_neighbors = min(10, max(1, X.shape[0] // 10))
        relief = ReliefF(n_features_to_select=k_eff, n_neighbors=n_neighbors)
        relief.fit(X.to_numpy(), y.to_numpy())
        order = np.argsort(relief.feature_importances_)[::-1][:k_eff]
        return [int(i) for i in order]


@dataclass
class RFEBaseline:
    """Recursive feature elimination over a random forest (wrapper baseline)."""

    name: str

    def select(self, X: pd.DataFrame, y: pd.Series, task: str, k: int) -> list[int]:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import RFE

        k_eff = min(k, X.shape[1])
        estimator = (
            RandomForestClassifier(n_estimators=100, random_state=0)
            if task == "classification"
            else RandomForestRegressor(n_estimators=100, random_state=0)
        )
        rfe = RFE(estimator=estimator, n_features_to_select=k_eff, step=0.2)
        rfe.fit(X.to_numpy(), y.to_numpy())
        return [int(i) for i in np.flatnonzero(rfe.get_support())]


@dataclass
class SkfeatureBaseline:
    """CMIM / JMI conditional-MI comparators via skfeature; self-disables if absent.

    The pairwise MRMR engine cannot express conditional redundancy (documented
    limitation), so CMIM/JMI are provided only through this external adapter.
    """

    name: str
    method: str  # "CMIM" | "JMI"

    @staticmethod
    def available() -> bool:
        import importlib.util

        return importlib.util.find_spec("skfeature") is not None

    def select(self, X: pd.DataFrame, y: pd.Series, task: str, k: int) -> list[int]:
        if task != "classification":
            raise ValueError(
                f"{self.method} is a classification-only conditional-MI method; "
                f"got task={task!r}. Drop this cell from regression grids."
            )
        if not self.available():
            raise RuntimeError(
                f"{self.method} unavailable: install skfeature-chappers into the "
                f"benchmarks group, or drop this cell from the grid."
            )
        from skfeature.function.information_theoretical_based import CIFE, CMIM, DISR, ICAP, JMI

        impl = {
            "CMIM": CMIM.cmim,
            "JMI": JMI.jmi,
            "CIFE": CIFE.cife,
            "ICAP": ICAP.icap,
            "DISR": DISR.disr,
        }[self.method]
        k_eff = min(k, X.shape[1])
        # skfeature-chappers' cmim/jmi return a single ndarray of selected indices
        # (already length n_selected_features), not the legacy (order, _, _) tuple.
        order = np.asarray(impl(X.to_numpy(), y.to_numpy().astype(int), n_selected_features=k_eff))
        return [int(i) for i in order.ravel()[:k_eff]]


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #
CELLS: dict[str, SelectorAdapter] = {
    # Literal information-theoretic criteria (MI relevance AND MI redundancy),
    # matching Brown 2012's (beta, gamma) linear family:
    #   MID = mRMR difference (Peng 2005),  MIQ = mRMR quotient (Peng 2005),
    #   MIFS = Battiti 1994 (beta=1, i.e. unweighted sum redundancy).
    "MID": MRMRCellAdapter("MID", "mi", "mutual_info_sklearn", "difference", "mean"),
    "MIQ": MRMRCellAdapter("MIQ", "mi", "mutual_info_sklearn", "quotient", "mean"),
    "MIFS": MRMRCellAdapter("MIFS", "mi", "mutual_info_sklearn", "difference", "sum"),
    # F-statistic relevance + |corr| redundancy family (Zhao 2019 FCD/FCQ):
    "FCD": MRMRCellAdapter("FCD", "f", "pearson_abs", "difference", "mean"),
    "FCQ": MRMRCellAdapter("FCQ", "f", "pearson_abs", "quotient", "mean"),
    "ModMRMR": MRMRCellAdapter("ModMRMR", "f", "pearson_abs", "multiplicative", "max"),
    "ModMRMR_mi": MRMRCellAdapter("ModMRMR_mi", "mi", "pearson_abs", "multiplicative", "max"),
    "ModMRMR_dcor": MRMRCellAdapter("ModMRMR_dcor", "f", "distance_corr", "multiplicative", "max"),
}

BASELINES: dict[str, SelectorAdapter] = {
    "skb_f": SelectKBestAdapter("skb_f", "f"),
    "skb_mi": SelectKBestAdapter("skb_mi", "mutual_info"),
    "mrmr_smazzanti": MRMRSelectionBaseline("mrmr_smazzanti"),
    "relieff": ReliefFBaseline("relieff"),
    "rfe": RFEBaseline("rfe"),
    "cmim": SkfeatureBaseline("cmim", "CMIM"),
    "jmi": SkfeatureBaseline("jmi", "JMI"),
    "cife": SkfeatureBaseline("cife", "CIFE"),
    "icap": SkfeatureBaseline("icap", "ICAP"),
    "disr": SkfeatureBaseline("disr", "DISR"),
}


def list_cells() -> list[str]:
    return list(CELLS) + list(BASELINES)


def get_adapter(name: str) -> SelectorAdapter:
    if name in CELLS:
        return CELLS[name]
    if name in BASELINES:
        return BASELINES[name]
    raise KeyError(f"Unknown adapter {name!r}; known: {list_cells()}")
