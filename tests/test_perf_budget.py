"""PBE-F30: run_lag_aware_mod_mrmr wall-clock budget.

Budget stored in tests/perf_budgets.yml under key PBE-F30.
Run with:
    uv run python -m pytest tests/test_perf_budget.py -m perf_budget -s
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from modmrmr.models import (
    LagAwareModMRMRConfig,
    PairwiseScorerSpec,
)
from modmrmr.tmrmr.run import run_lag_aware_mod_mrmr


def _load_budget(tag: str) -> float:
    """Load budget_seconds for a given PBE tag from perf_budgets.yml.

    Args:
        tag: PBE identifier string (e.g. 'PBE-F30').

    Returns:
        Budget in seconds as a float.
    """
    path = Path(__file__).parent / "perf_budgets.yml"
    with open(path, encoding="utf-8") as f:
        budgets: dict[str, dict[str, object]] = yaml.safe_load(f)
    return float(budgets[tag]["budget_seconds"])


def _make_scorer_spec(name: str = "spearman_abs") -> PairwiseScorerSpec:
    """Build a minimal scorer spec for perf benchmarking."""
    return PairwiseScorerSpec(
        name=name,
        backend="scipy",
        normalization="rank_percentile",
        significance_method="none",
    )


@pytest.mark.slow
@pytest.mark.perf_budget
def test_mod_mrmr_within_budget() -> None:
    """PBE-F30: run_lag_aware_mod_mrmr (p=200 covariates, N=500) <= budget.

    Generates a target series and 200 synthetic covariate series, each of
    length 500, then runs the full Lag-Aware ModMRMR pipeline.  Asserts
    the wall-clock time is within the budget registered in perf_budgets.yml
    under PBE-F30.
    """
    rng = np.random.default_rng(42)
    N = 500
    target = rng.standard_normal(N)
    covariates = {f"cov_{i:03d}": rng.standard_normal(N) for i in range(200)}

    config = LagAwareModMRMRConfig(
        forecast_horizon=1,
        candidate_lags=[1, 2, 3],
        max_selected_features=10,
        relevance_scorer=_make_scorer_spec("spearman_abs"),
        redundancy_scorer=_make_scorer_spec("spearman_abs"),
    )

    budget = _load_budget("PBE-F30")

    t0 = time.perf_counter()
    result = run_lag_aware_mod_mrmr(
        target=target,
        covariates=covariates,
        config=config,
        random_state=42,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n[PBE-F30] run_lag_aware_mod_mrmr elapsed: {elapsed:.3f}s  (budget: {budget}s)")
    assert result is not None
    assert elapsed <= budget, (
        f"run_lag_aware_mod_mrmr took {elapsed:.2f}s, exceeds budget {budget}s. "
        "If hardware has changed, re-measure and update perf_budgets.yml."
    )
