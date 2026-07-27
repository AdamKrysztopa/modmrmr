"""Tests for the profiling sweep driver."""

from __future__ import annotations

import csv

import pytest

from benchmarks.profiling.grid import ProfileCell, ProfileGrid
from benchmarks.profiling.run import (
    CSV_COLUMNS,
    load_completed_keys,
    project_seconds,
    run_cell,
    run_grid,
)


def test_project_seconds_fits_a_power_law():
    """Doubling n on an O(n^2) kernel must project a ~4x cost."""
    observed = [(100, 1.0), (200, 4.0)]
    assert project_seconds(observed, 400) == pytest.approx(16.0, rel=0.05)


def test_project_seconds_handles_linear_scaling():
    observed = [(100, 1.0), (200, 2.0)]
    assert project_seconds(observed, 400) == pytest.approx(4.0, rel=0.05)


def test_project_seconds_with_one_observation_assumes_linear():
    assert project_seconds([(100, 1.0)], 300) == pytest.approx(3.0, rel=0.05)


def test_load_completed_keys_on_missing_file_returns_empty(tmp_path):
    assert load_completed_keys(tmp_path / "absent.csv") == set()


def test_load_completed_keys_reads_back_written_rows(tmp_path):
    path = tmp_path / "out.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                **{c: "" for c in CSV_COLUMNS},
                "benchmark": "scorer_scaling",
                "scorer": "pearson_abs",
                "n": 100,
                "p": 0,
                "data_kind": "continuous",
                "status": "ok",
            }
        )
    assert load_completed_keys(path) == {("scorer_scaling", "pearson_abs", 100, 0, "continuous")}


def test_run_cell_produces_a_complete_row():
    cell = ProfileCell(
        benchmark="scorer_scaling",
        scorer="pearson_abs",
        n=200,
        p=0,
        data_kind="continuous",
        seed=1,
    )
    row = run_cell(cell, repeats=2, warmup=0)
    assert set(row) == set(CSV_COLUMNS)
    assert row["status"] == "ok"
    assert float(row["median_s"]) > 0.0


def test_run_grid_writes_one_row_per_cell(tmp_path):
    grid = ProfileGrid(
        benchmark="scorer_scaling",
        scorers=["pearson_abs"],
        n_values=[100, 200],
        p_values=[0],
        data_kinds=["continuous"],
    )
    out = tmp_path / "sweep.csv"
    run_grid(grid, out, budget_s=60.0, repeats=1, warmup=0)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2


def test_run_grid_resumes_and_does_not_duplicate(tmp_path):
    grid = ProfileGrid(
        benchmark="scorer_scaling",
        scorers=["pearson_abs"],
        n_values=[100, 200],
        p_values=[0],
        data_kinds=["continuous"],
    )
    out = tmp_path / "sweep.csv"
    run_grid(grid, out, budget_s=60.0, repeats=1, warmup=0)
    run_grid(grid, out, budget_s=60.0, repeats=1, warmup=0)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2


def test_cost_guard_records_a_skipped_row_rather_than_omitting_it(tmp_path):
    """A skipped cell and an absent cell must be distinguishable in the CSV.

    Silent truncation reads as 'we measured everything' when we did not.
    """
    grid = ProfileGrid(
        benchmark="scorer_scaling",
        scorers=["pearson_abs"],
        n_values=[200, 100000],
        p_values=[0],
        data_kinds=["continuous"],
    )
    out = tmp_path / "sweep.csv"
    run_grid(grid, out, budget_s=1e-9, repeats=1, warmup=0)
    rows = list(csv.DictReader(out.open()))
    statuses = {r["n"]: r["status"] for r in rows}
    assert statuses["200"] == "ok"
    assert statuses["100000"] == "skipped_projected_cost"
    skipped = next(r for r in rows if r["n"] == "100000")
    assert float(skipped["projected_s"]) > 0.0


def test_cost_guard_projects_per_cell_wall_clock(tmp_path):
    """budget_s is a per-CELL wall-clock budget, not a per-repeat one.

    A cell whose projected per-repeat median is under budget but whose full
    run (repeats + warmup) exceeds it must be skipped. With one observation
    the projection is linear: (100, 0.2) -> n=200 projects 0.4s per repeat,
    i.e. 2.4s per cell at repeats=5, warmup=1 — over the 1.0s budget, while
    the bare per-repeat 0.4s would pass.
    """
    grid = ProfileGrid(
        benchmark="scorer_scaling",
        scorers=["pearson_abs"],
        n_values=[100, 200],
        p_values=[0],
        data_kinds=["continuous"],
    )
    out = tmp_path / "sweep.csv"
    out.write_text(
        ",".join(CSV_COLUMNS)
        + "\n"
        + "scorer_scaling,pearson_abs,100,0,continuous,0,ok,0.2,0.01,1,\n"
    )
    run_grid(grid, out, budget_s=1.0, repeats=5, warmup=1)
    rows = list(csv.DictReader(out.open()))
    new_row = next(r for r in rows if r["n"] == "200")
    assert new_row["status"] == "skipped_projected_cost"
    assert float(new_row["projected_s"]) > 1.0


def test_resume_seeds_the_cost_guard_from_prior_rows(tmp_path):
    """After an interruption the guard must not restart cold: observations are
    rebuilt from the CSV's ok rows, so the first remaining cell per scorer is
    still guarded."""
    grid = ProfileGrid(
        benchmark="scorer_scaling",
        scorers=["pearson_abs"],
        n_values=[100, 200],
        p_values=[0],
        data_kinds=["continuous"],
    )
    out = tmp_path / "sweep.csv"
    out.write_text(
        ",".join(CSV_COLUMNS)
        + "\n"
        + "scorer_scaling,pearson_abs,100,0,continuous,0,ok,0.2,0.01,1,\n"
    )
    # Linear projection from the seeded row: 0.4s/repeat at n=200, i.e. 0.4s
    # per cell at repeats=1, warmup=0 — over a 0.3s budget ONLY if seeded.
    run_grid(grid, out, budget_s=0.3, repeats=1, warmup=0)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2  # the seeded row is completed, never re-attempted
    new_row = next(r for r in rows if r["n"] == "200")
    assert new_row["status"] == "skipped_projected_cost"
