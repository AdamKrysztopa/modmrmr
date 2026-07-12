"""Full-factorial ``SelectorSpec`` -> ``MRMRSelector`` builder.

The mechanism-suite runner (``mechanism.protocol``) grades named *cells*
(``benchmarks.cells.CELLS``), which confounds operator, aggregation, relevance
family, and redundancy scorer into a handful of hand-picked combinations
(e.g. ``ModMRMR`` vs ``FCQ`` differ in operator+aggregation *and* nothing
else, but ``MID`` vs ``FCD`` differ in relevance family *and* redundancy
scorer at once). This module instead enumerates the full 4-axis cross
product so Phase B/C/D of the design-space study can hold three axes fixed
and vary the fourth cleanly.

``SelectorSpec`` is a small immutable DTO (frozen dataclass, per the
project's DTO convention); ``build_selector`` resolves its ``relevance_family``
to a concrete, task-specific scorer name and constructs an ``MRMRSelector``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from modmrmr.core.estimator import MRMRSelector

RELEVANCE_FAMILIES = ("pearson", "spearman", "f", "mi", "dcor")
REDUNDANCY_SCORERS = ("pearson_abs", "spearman_abs", "mutual_info_sklearn", "distance_corr")
OPERATORS = ("difference", "quotient", "multiplicative")
AGGREGATIONS = ("mean", "max", "sum")

# relevance_family x task -> concrete relevance name accepted by MRMRSelector.
# Empirically verified (see task-B-report.md): "f" and "mi" are genuinely
# task-specific (f_classif/mutual_info_classif raise on a continuous target,
# mutual_info_classif raises on a continuous target); "pearson"/"spearman"/
# "dcor" resolve to the same registered scorer name for both tasks.
_RELEVANCE_BY_FAMILY: dict[tuple[str, str], str] = {
    ("pearson", "classification"): "pearson_abs",
    ("pearson", "regression"): "pearson_abs",
    ("spearman", "classification"): "spearman_abs",
    ("spearman", "regression"): "spearman_abs",
    ("f", "classification"): "f_classif",
    ("f", "regression"): "f_regression",
    ("mi", "classification"): "mutual_info_classif",
    ("mi", "regression"): "mutual_info_regression",
    ("dcor", "classification"): "distance_corr",
    ("dcor", "regression"): "distance_corr",
}

_VALID_TASKS = ("classification", "regression")


@dataclass(frozen=True)
class SelectorSpec:
    """One point in the (relevance_family x redundancy x operator x aggregation) grid.

    ``relevance_family`` is a family label (e.g. ``"f"``, ``"mi"``), not a concrete
    scorer name -- :func:`build_selector` resolves it per-task via
    ``_RELEVANCE_BY_FAMILY``. ``redundancy`` is already a concrete, registered
    redundancy-scorer name (redundancy scoring has no task-dependent variant).
    """

    relevance_family: str
    redundancy: str
    operator: str
    aggregation: str

    @property
    def label(self) -> str:
        return f"{self.relevance_family}|{self.redundancy}|{self.operator}|{self.aggregation}"


FULL_FACTORIAL: tuple[SelectorSpec, ...] = tuple(
    SelectorSpec(relevance_family=rf, redundancy=rd, operator=op, aggregation=ag)
    for rf, rd, op, ag in product(RELEVANCE_FAMILIES, REDUNDANCY_SCORERS, OPERATORS, AGGREGATIONS)
)


def _resolve_relevance(relevance_family: str, task: str) -> str:
    """Resolve ``relevance_family`` to a concrete ``MRMRSelector(relevance=...)`` name.

    Raises ``ValueError`` for an unknown family or a task outside
    ``{"classification", "regression"}`` (callers must resolve ``"auto"`` to a
    concrete task before calling ``build_selector``).
    """
    if task not in _VALID_TASKS:
        raise ValueError(
            f"build_selector requires a resolved task in {_VALID_TASKS}; got {task!r}. "
            "Resolve 'auto' via modmrmr.core.task.detect_task before calling build_selector."
        )
    try:
        return _RELEVANCE_BY_FAMILY[(relevance_family, task)]
    except KeyError:
        valid = sorted({rf for rf, _ in _RELEVANCE_BY_FAMILY})
        raise ValueError(
            f"Unknown relevance family {relevance_family!r}; expected one of {valid}"
        ) from None


def build_selector(
    spec: SelectorSpec,
    task: str,
    n_features: int | None,
    score_threshold: float | None,
    random_state: int,
) -> MRMRSelector:
    """Build an ``MRMRSelector`` from ``spec``, resolving relevance for ``task``."""
    relevance = _resolve_relevance(spec.relevance_family, task)
    return MRMRSelector(
        n_features=n_features,
        relevance=relevance,
        redundancy=spec.redundancy,
        operator=spec.operator,
        aggregation=spec.aggregation,
        task=task,
        random_state=random_state,
        score_threshold=score_threshold,
    )


# Readable named methods for reporting, matching benchmarks/cells.py's canonical
# cells (MID/MIQ/FCD/FCQ/ModMRMR) but expressed as SelectorSpecs.
CANONICAL_NAMED: dict[str, SelectorSpec] = {
    "MID": SelectorSpec("mi", "mutual_info_sklearn", "difference", "mean"),
    "MIQ": SelectorSpec("mi", "mutual_info_sklearn", "quotient", "mean"),
    "FCD": SelectorSpec("f", "pearson_abs", "difference", "mean"),
    "FCQ": SelectorSpec("f", "pearson_abs", "quotient", "mean"),
    "ModMRMR": SelectorSpec("f", "pearson_abs", "multiplicative", "max"),
}
