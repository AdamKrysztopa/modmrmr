"""Scorer library: Protocol, registry, adapters, and built-in families.

Plan B measure boundary
-----------------------
CMIM and JMI are *conditional* mutual-information criteria: they score a
candidate feature's redundancy conditioned on the already-selected set, which
is stateful and NOT expressible through the stateless pairwise
``PairwiseDependenceScorer.score_pair(x, y)`` Protocol. They are therefore NOT
registered here; they are adapted as external named comparators in Plan C
(``benchmarks/``). All Plan B additions (``f_regression``, ``f_classif``,
``mutual_info_classif``, ``distance_corr``, ``rdc``, ``relieff``) ARE genuinely
pairwise and live in this registry.
"""

from __future__ import annotations

from modmrmr.core.normalize import build_normalizer
from modmrmr.core.scorers.base import (
    _MIN_PAIRS as _MIN_PAIRS,
)
from modmrmr.core.scorers.base import (
    PairwiseDependenceScorer,
    as_importance_function,
    as_penalty_matrix,
    build_scorer,
    fast_pearson_penalty,
    get_scorer,
    list_scorers,
    make_diagnostics,
    register_scorer,
)
from modmrmr.core.scorers.base import (
    _RawScore as _RawScore,
)
from modmrmr.core.scorers.bspline_mi import _BSplineMiScorer
from modmrmr.core.scorers.copula_mi import _CopulaMiScorer
from modmrmr.core.scorers.dependence import _DistanceCorrScorer, _RdcScorer
from modmrmr.core.scorers.kde_mi import _KdeMiScorer
from modmrmr.core.scorers.linear import _PearsonAbsScorer, _SpearmanAbsScorer
from modmrmr.core.scorers.mutual_info import (
    _AmiAdaptiveScorer,
    _CattKnnMiScorer,
    _GcmiScorer,
    _MixedKsgScorer,
    _MutualInfoSklearnScorer,
)
from modmrmr.core.scorers.relief import _ReliefFScorer
from modmrmr.core.scorers.statistical import (
    _FClassifScorer,
    _FRegressionScorer,
    _MutualInfoClassifScorer,
)
from modmrmr.core.scorers.tree import _TreeR2Scorer

register_scorer("pearson_abs", _PearsonAbsScorer())
register_scorer("spearman_abs", _SpearmanAbsScorer())
register_scorer("mutual_info_sklearn", _MutualInfoSklearnScorer())
register_scorer("catt_knn_mi", _CattKnnMiScorer())
register_scorer("ksg_mi", _CattKnnMiScorer())
register_scorer("cross_ami_score", _CattKnnMiScorer())
register_scorer("gcmi", _GcmiScorer())
register_scorer("mixed_ksg", _MixedKsgScorer())
register_scorer("ami_adaptive", _AmiAdaptiveScorer())
register_scorer("bspline_mi", _BSplineMiScorer())
register_scorer("kde_mi", _KdeMiScorer())
register_scorer("copula_mi", _CopulaMiScorer())
register_scorer("tree_r2", _TreeR2Scorer())
register_scorer("f_regression", _FRegressionScorer())
register_scorer("f_classif", _FClassifScorer())
register_scorer("mutual_info_classif", _MutualInfoClassifScorer())
register_scorer("distance_corr", _DistanceCorrScorer())
register_scorer("rdc", _RdcScorer())
register_scorer("relieff", _ReliefFScorer())

for _n in (3, 5, 8, 16):
    register_scorer(f"mi_classif_k{_n}", _MutualInfoClassifScorer().with_neighbors(_n))
    register_scorer(f"mi_reg_k{_n}", _MutualInfoSklearnScorer().with_neighbors(_n))
register_scorer("ksg_k16", _CattKnnMiScorer().with_neighbors(16))
del _n

__all__ = [
    "PairwiseDependenceScorer",
    "build_scorer",
    "build_normalizer",
    "make_diagnostics",
    "register_scorer",
    "get_scorer",
    "list_scorers",
    "as_importance_function",
    "as_penalty_matrix",
    "fast_pearson_penalty",
]
