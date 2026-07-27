"""Tests for the profiling grid specification."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmarks.profiling.grid import DEFAULT_GRIDS, ProfileCell, ProfileGrid


def test_grid_expands_to_cartesian_product():
    grid = ProfileGrid(
        benchmark="scorer_scaling",
        scorers=["pearson_abs", "gcmi"],
        n_values=[100, 200],
        p_values=[0],
        data_kinds=["continuous", "discrete"],
    )
    cells = grid.cells()
    assert len(cells) == 2 * 2 * 1 * 2


def test_cell_seeds_are_distinct_and_deterministic():
    grid = ProfileGrid(
        benchmark="scorer_scaling",
        scorers=["pearson_abs", "gcmi"],
        n_values=[100],
        p_values=[0],
        data_kinds=["continuous"],
    )
    seeds = [c.seed for c in grid.cells()]
    assert len(set(seeds)) == len(seeds)
    assert [c.seed for c in grid.cells()] == seeds


def test_cell_key_identifies_the_cell_without_the_seed():
    """Resume matches on identity, not on seed, so the key must exclude it."""
    cell = ProfileCell(
        benchmark="scorer_scaling",
        scorer="gcmi",
        n=100,
        p=0,
        data_kind="continuous",
        seed=5,
    )
    assert cell.key == ("scorer_scaling", "gcmi", 100, 0, "continuous")


def test_cell_is_frozen():
    cell = ProfileCell(benchmark="b", scorer="s", n=1, p=0, data_kind="continuous", seed=0)
    with pytest.raises(ValidationError):
        cell.n = 2


def test_default_grids_cover_all_three_benchmarks():
    assert set(DEFAULT_GRIDS) == {"scorer_scaling", "driver_scaling", "end_to_end"}


def test_scorer_scaling_grid_includes_the_discrete_kind():
    """Without discrete data the O(n^2) tie loop never executes."""
    assert "discrete" in DEFAULT_GRIDS["scorer_scaling"].data_kinds


def test_driver_scaling_grid_holds_the_scorer_fixed():
    """The driver benchmark isolates p^2 dispatch overhead from scorer cost."""
    assert DEFAULT_GRIDS["driver_scaling"].scorers == ["pearson_abs"]
    assert len(DEFAULT_GRIDS["driver_scaling"].p_values) >= 3


def test_expensive_scorers_are_swept_only_at_the_smallest_n():
    """tree_r2 fits a 200-tree forest and relieff fits skrebate at every pair;
    at large n they would dominate the sweep's wall-clock without informing
    the Rust gate (the code-ownership filter excludes them anyway)."""
    cells = DEFAULT_GRIDS["scorer_scaling"].cells()
    for scorer in ("tree_r2", "relieff"):
        ns = {c.n for c in cells if c.scorer == scorer}
        assert ns == {200}, f"{scorer} swept beyond smallest n: {ns}"
    cheap_ns = {c.n for c in cells if c.scorer == "pearson_abs"}
    assert max(cheap_ns) == 10000
