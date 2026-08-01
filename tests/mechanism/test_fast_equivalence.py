"""Equivalence gates for the memoized and reference factorial-grid runners.

The fast path (``mechanism.fast_factorial_protocol``) memoizes the relevance
vector and redundancy matrix per ``(dataset, seed, split-role, relevance,
redundancy)`` instead of recomputing them for every ``(spec, k, threshold)``.
That is a pure performance change -- the greedy selection every row performs
must be numerically identical to the reference oracle
(``mechanism.factorial_protocol.run_factorial_grid``). If this test fails, the
memoization is wrong: it must be fixed until identical, never shipped
divergent.

The default suite keeps a compact exact-parity smoke covering every operator,
both principal aggregation paths, representative relevance/redundancy scorer
families, and cache reuse across operators/aggregations. The broader two-dataset
parameter grid and its relative timing check are ``perf_budget`` tests, intended
for the serial performance lane.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from mechanism.factorial import SelectorSpec
from mechanism.factorial_protocol import run_factorial_grid
from mechanism.fast_factorial_protocol import run_fast_factorial_grid

_SMOKE_SPECS = [
    SelectorSpec("f", "pearson_abs", "difference", "mean"),
    SelectorSpec("f", "pearson_abs", "quotient", "max"),
    SelectorSpec("mi", "mutual_info_sklearn", "multiplicative", "mean"),
]
_SMOKE_DATASETS = ["quotient_trap_reg"]
_SMOKE_KS = [3]
_SMOKE_THRESHOLDS = [0.05]
_SMOKE_SEEDS = [0]

_BROAD_SPECS = [
    # relevance=f, redundancy=pearson_abs, operator=difference, agg=mean
    SelectorSpec("f", "pearson_abs", "difference", "mean"),
    # relevance=f, redundancy=pearson_abs, operator=quotient, agg=max -- same
    # measure pair as above, different operator/aggregation (memoization payoff).
    SelectorSpec("f", "pearson_abs", "quotient", "max"),
    # relevance=f, redundancy=pearson_abs, operator=multiplicative, agg=mean
    SelectorSpec("f", "pearson_abs", "multiplicative", "mean"),
    # relevance=mi, redundancy=mutual_info_sklearn, operator=difference, agg=max
    SelectorSpec("mi", "mutual_info_sklearn", "difference", "max"),
    # relevance=pearson, redundancy=distance_corr, operator=quotient, agg=mean
    SelectorSpec("pearson", "distance_corr", "quotient", "mean"),
]
_BROAD_DATASETS = ["quotient_trap_reg", "radial"]
_BROAD_KS = [1, 2, 3, 5]
_BROAD_THRESHOLDS = [0.0, 0.05, 0.1]
_BROAD_SEEDS = [0, 1]

_SORT_KEYS = ["dataset", "spec", "stop_mode", "k", "score_threshold", "seed"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(_SORT_KEYS, kind="stable").reset_index(drop=True)


def _assert_grids_equal(reference: pd.DataFrame, fast: pd.DataFrame) -> None:
    ref_sorted = _sorted(reference).drop(columns=["runtime_s"])
    fast_sorted = _sorted(fast).drop(columns=["runtime_s"])

    pd.testing.assert_frame_equal(ref_sorted, fast_sorted)


def test_fast_grid_smoke_is_byte_identical_to_reference_oracle() -> None:
    reference = run_factorial_grid(
        _SMOKE_SPECS, _SMOKE_DATASETS, _SMOKE_KS, _SMOKE_THRESHOLDS, _SMOKE_SEEDS
    )
    fast = run_fast_factorial_grid(
        _SMOKE_SPECS, _SMOKE_DATASETS, _SMOKE_KS, _SMOKE_THRESHOLDS, _SMOKE_SEEDS
    )

    _assert_grids_equal(reference, fast)


@pytest.mark.perf_budget
def test_fast_grid_broad_is_byte_identical_to_reference_oracle() -> None:
    reference = run_factorial_grid(
        _BROAD_SPECS, _BROAD_DATASETS, _BROAD_KS, _BROAD_THRESHOLDS, _BROAD_SEEDS
    )
    fast = run_fast_factorial_grid(
        _BROAD_SPECS, _BROAD_DATASETS, _BROAD_KS, _BROAD_THRESHOLDS, _BROAD_SEEDS
    )

    _assert_grids_equal(reference, fast)


@pytest.mark.perf_budget
def test_fast_grid_broad_is_actually_faster() -> None:
    t0 = time.perf_counter()
    run_factorial_grid(_BROAD_SPECS, _BROAD_DATASETS, _BROAD_KS, _BROAD_THRESHOLDS, _BROAD_SEEDS)
    reference_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_fast_factorial_grid(
        _BROAD_SPECS, _BROAD_DATASETS, _BROAD_KS, _BROAD_THRESHOLDS, _BROAD_SEEDS
    )
    fast_s = time.perf_counter() - t0

    assert fast_s < reference_s, (
        f"expected the memoized fast grid to be faster on this grid; "
        f"reference={reference_s:.2f}s fast={fast_s:.2f}s"
    )
