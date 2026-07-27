"""Grid specification for the profiling sweep.

Three benchmarks answer three different questions and must not be conflated:

- ``scorer_scaling`` varies n per scorer and yields the fitted scaling
  exponent, the primary input to the Rust decision gate.
- ``driver_scaling`` varies p with a fixed cheap scorer, isolating the O(p^2)
  pairwise-dispatch overhead from the cost of the scorer itself.
- ``end_to_end`` measures full selector fits and supplies the Amdahl
  denominator.
"""

from __future__ import annotations

from itertools import product

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.profiling.data import DataKind


class ProfileCell(BaseModel):
    """One measured point in the sweep."""

    model_config = ConfigDict(frozen=True)

    benchmark: str
    scorer: str
    n: int
    # 0 for per-pair benchmarks, where feature count is not a factor.
    p: int
    data_kind: DataKind
    seed: int

    @property
    def key(self) -> tuple[str, str, int, int, str]:
        """Identity of this cell for resume matching (seed excluded).

        The seed is derived from position and would change if the grid were
        reordered; including it would defeat resume across grid edits.
        """
        return (self.benchmark, self.scorer, self.n, self.p, self.data_kind)


class ProfileGrid(BaseModel):
    """A cartesian sweep specification."""

    model_config = ConfigDict(frozen=True)

    benchmark: str
    scorers: list[str]
    n_values: list[int]
    p_values: list[int] = Field(default_factory=lambda: [0])
    data_kinds: list[DataKind]
    base_seed: int = 20260719
    # Per-scorer cap on n: third-party-fit-bound scorers (a fitted model per
    # pair) are swept only at small n, or they dominate the sweep wall-clock.
    max_n_per_scorer: dict[str, int] = Field(default_factory=dict)

    def cells(self) -> list[ProfileCell]:
        """Expand to the cartesian product, capped per scorer, seeded by position."""
        full_n = max(self.n_values)
        combos = (
            (scorer, n, p, kind)
            for scorer, n, p, kind in product(
                self.scorers, self.n_values, self.p_values, self.data_kinds
            )
            if n <= self.max_n_per_scorer.get(scorer, full_n)
        )
        return [
            ProfileCell(
                benchmark=self.benchmark,
                scorer=scorer,
                n=n,
                p=p,
                data_kind=kind,
                seed=self.base_seed + idx,
            )
            for idx, (scorer, n, p, kind) in enumerate(combos)
        ]


# All registered scorers except the two that are third-party-fit-bound at every
# pair (tree_r2 fits a 200-tree forest, relieff fits skrebate). Those two are
# still swept, but only at the smallest n, via the max_n_per_scorer cap below.
_ALL_SCORERS = [
    "pearson_abs",
    "spearman_abs",
    "gcmi",
    "copula_mi",
    "mutual_info_sklearn",
    "catt_knn_mi",
    "mixed_ksg",
    "ami_adaptive",
    "bspline_mi",
    "kde_mi",
    "distance_corr",
    "rdc",
]
# Third-party-fit-bound scorers: swept at the smallest n only, so their
# per-pair model fits do not dominate the sweep wall-clock. The Rust gate's
# code-ownership filter excludes them regardless; one n point documents their
# absolute cost.
_EXPENSIVE_SCORERS = ["tree_r2", "relieff"]

DEFAULT_GRIDS: dict[str, ProfileGrid] = {
    "scorer_scaling": ProfileGrid(
        benchmark="scorer_scaling",
        scorers=_ALL_SCORERS + _EXPENSIVE_SCORERS,
        n_values=[200, 500, 1000, 2000, 5000, 10000],
        p_values=[0],
        data_kinds=["continuous", "discrete", "mixed"],
        max_n_per_scorer={s: 200 for s in _EXPENSIVE_SCORERS},
    ),
    "driver_scaling": ProfileGrid(
        benchmark="driver_scaling",
        scorers=["pearson_abs"],
        n_values=[500],
        p_values=[25, 50, 100, 200, 400],
        data_kinds=["continuous"],
    ),
    "end_to_end": ProfileGrid(
        benchmark="end_to_end",
        scorers=["pearson_abs", "gcmi", "mixed_ksg", "mutual_info_sklearn"],
        n_values=[500, 2000],
        p_values=[50, 200],
        data_kinds=["continuous", "discrete"],
    ),
}
