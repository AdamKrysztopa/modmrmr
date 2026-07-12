"""Hard equivalence gate: ``run_fast_factorial_grid`` must byte-match ``run_factorial_grid``.

The fast path (``mechanism.fast_factorial_protocol``) memoizes the relevance
vector and redundancy matrix per ``(dataset, seed, split-role, relevance,
redundancy)`` instead of recomputing them for every ``(spec, k, threshold)``.
That is a pure performance change -- the greedy selection every row performs
must be numerically identical to the reference oracle
(``mechanism.factorial_protocol.run_factorial_grid``). If this test fails, the
memoization is wrong: it must be fixed until identical, never shipped
divergent.

The grid below is representative, not exhaustive: 2 datasets (including
``quotient_trap_reg``), >=4 specs spanning >=2 relevance families and >=2
redundancy scorers and all 3 operators + agg {mean, max}, ks=[1,2,3,5],
thresholds=[0.0,0.05,0.1], seeds=[0,1].
"""

from __future__ import annotations

import time

import pandas as pd

from mechanism.factorial import SelectorSpec
from mechanism.factorial_protocol import run_factorial_grid
from mechanism.fast_factorial_protocol import run_fast_factorial_grid

_SPECS = [
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
_DATASETS = ["quotient_trap_reg", "radial"]
_KS = [1, 2, 3, 5]
_THRESHOLDS = [0.0, 0.05, 0.1]
_SEEDS = [0, 1]

_SORT_KEYS = ["dataset", "spec", "stop_mode", "k", "score_threshold", "seed"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(_SORT_KEYS, kind="stable").reset_index(drop=True)


def test_fast_grid_is_byte_identical_to_reference_oracle() -> None:
    reference = run_factorial_grid(_SPECS, _DATASETS, _KS, _THRESHOLDS, _SEEDS)
    fast = run_fast_factorial_grid(_SPECS, _DATASETS, _KS, _THRESHOLDS, _SEEDS)

    ref_sorted = _sorted(reference).drop(columns=["runtime_s"])
    fast_sorted = _sorted(fast).drop(columns=["runtime_s"])

    pd.testing.assert_frame_equal(ref_sorted, fast_sorted)


def test_fast_grid_is_actually_faster() -> None:
    t0 = time.perf_counter()
    run_factorial_grid(_SPECS, _DATASETS, _KS, _THRESHOLDS, _SEEDS)
    reference_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_fast_factorial_grid(_SPECS, _DATASETS, _KS, _THRESHOLDS, _SEEDS)
    fast_s = time.perf_counter() - t0

    assert fast_s < reference_s, (
        f"expected the memoized fast grid to be faster on this grid; "
        f"reference={reference_s:.2f}s fast={fast_s:.2f}s"
    )
