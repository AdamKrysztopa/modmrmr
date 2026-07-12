"""Ground-truth feature mask carried by every mechanism dataset.

Three disjoint roles that partition all columns: ``informative`` (truly drive y),
``codependent`` (grouped; each group is redundant with an informative signal), and
``noise`` (pure junk). Recovery scoring reads this to reward picking distinct
informative signal and penalise picking redundant duplicates or noise.
"""

from __future__ import annotations

from dataclasses import dataclass

_DEPENDENCE = ("linear", "nonlinear", "mixed")


@dataclass(frozen=True)
class GroundTruth:
    informative: tuple[int, ...]
    codependent: tuple[tuple[int, ...], ...]
    noise: tuple[int, ...]
    dependence: str
    n_features: int

    def __post_init__(self) -> None:
        if self.dependence not in _DEPENDENCE:
            raise ValueError(f"dependence must be one of {_DEPENDENCE}, got {self.dependence!r}")
        flat = list(self.informative) + [i for g in self.codependent for i in g] + list(self.noise)
        if sorted(flat) != list(range(self.n_features)):
            raise ValueError(
                f"informative/codependent/noise must partition 0..{self.n_features - 1} "
                f"exactly once; got {sorted(flat)}"
            )

    @property
    def relevant_columns(self) -> frozenset[int]:
        return frozenset(self.informative) | frozenset(i for g in self.codependent for i in g)

    def codependent_group_of(self, idx: int) -> int | None:
        for gi, group in enumerate(self.codependent):
            if idx in group:
                return gi
        return None
