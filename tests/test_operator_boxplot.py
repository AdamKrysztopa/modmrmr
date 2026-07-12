"""Tests for the M5 matched-settings operator boxplot (analysis.operator_boxplot).

Smoke (figure PDF produced) + determinism (two stats computations identical).
Requires the committed results artifacts; skipped when they are absent so the
suite stays runnable on a fresh clone without the study outputs.
"""

from pathlib import Path

import pandas as pd
import pytest

FACTORIAL = Path("results/factorial.parquet")
REGQ_CACHE = Path("results/factorial_regquotient_golden.parquet")

pytestmark = pytest.mark.skipif(
    not (FACTORIAL.exists() and REGQ_CACHE.exists()),
    reason="results parquet artifacts not present",
)


@pytest.fixture(scope="module")
def per_run() -> pd.DataFrame:
    from analysis.operator_boxplot import matched_per_run

    return matched_per_run()


class TestMatchedPerRun:
    def test_four_arms_150_points_each(self, per_run):
        from analysis.operator_boxplot import ARMS

        counts = per_run.groupby("operator").size()
        assert sorted(counts.index) == sorted(ARMS)
        for arm in ARMS:
            assert counts[arm] == 150  # 15 golden datasets x 10 seeds

    def test_metrics_in_range(self, per_run):
        assert per_run["f1"].between(0.0, 1.0).all()
        assert per_run["noise_rate"].between(0.0, 1.0).all()


class TestStats:
    def test_stats_deterministic(self, per_run):
        from analysis.operator_boxplot import compute_stats

        a = compute_stats(per_run)
        b = compute_stats(per_run)
        pd.testing.assert_frame_equal(a, b)

    def test_stats_shape_and_columns(self, per_run):
        from analysis.operator_boxplot import compute_stats

        stats = compute_stats(per_run)
        # per metric: 1 Friedman row + C(4,2)=6 pairwise rows
        assert len(stats) == 2 * (1 + 6)
        assert list(stats.columns) == [
            "metric",
            "test",
            "arm_a",
            "arm_b",
            "statistic",
            "p_raw",
            "p_holm",
            "n_blocks",
        ]
        assert (stats["n_blocks"] == 15).all()

    def test_holm_at_least_raw(self, per_run):
        from analysis.operator_boxplot import compute_stats

        stats = compute_stats(per_run)
        pw = stats[stats["test"] == "wilcoxon_holm"]
        assert (pw["p_holm"] >= pw["p_raw"] - 1e-12).all()


class TestFigure:
    def test_figure_smoke(self, per_run, tmp_path, monkeypatch):
        import analysis.operator_boxplot as mod

        out = tmp_path / "fig_operator_boxplot.pdf"
        monkeypatch.setattr(mod, "ARTIFACTS", tmp_path)
        monkeypatch.setattr(mod, "FIGURE_OUT", out)
        path = mod.fig_operator_boxplot(per_run)
        assert path == out
        assert out.exists()
        assert out.stat().st_size > 1000  # a real PDF, not an empty stub
