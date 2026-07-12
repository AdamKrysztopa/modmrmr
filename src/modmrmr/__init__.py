"""ModMRMR: injectable maximum-relevance minimum-redundancy feature selection."""

from __future__ import annotations

from modmrmr.core import MRMR, ModMRMR, MRMRSelector, pearson_corr
from modmrmr.core.scorers import (
    as_importance_function,
    as_penalty_matrix,
    build_scorer,
    get_scorer,
    list_scorers,
    register_scorer,
)
from modmrmr.models import (
    LagAwareModMRMRConfig,
    LagAwareModMRMRResult,
    PairwiseScorerSpec,
)
from modmrmr.tmrmr import run_lag_aware_mod_mrmr, run_tmrmr

__version__ = "0.1.0"

__all__ = [
    "MRMR",
    "ModMRMR",
    "MRMRSelector",
    "pearson_corr",
    "run_tmrmr",
    "run_lag_aware_mod_mrmr",
    "register_scorer",
    "get_scorer",
    "list_scorers",
    "build_scorer",
    "as_importance_function",
    "as_penalty_matrix",
    "LagAwareModMRMRConfig",
    "LagAwareModMRMRResult",
    "PairwiseScorerSpec",
    "__version__",
]
