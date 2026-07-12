"""Tests for paper LaTeX table generation."""

from pathlib import Path

import pytest

FACTORIAL = Path("results/factorial.parquet")
BASELINES = Path("results/baseline_comparison_summary.csv")

pytestmark = pytest.mark.skipif(
    not (FACTORIAL.exists() and BASELINES.exists()), reason="results files absent"
)


def test_t2_named_methods_marks_classification_only(tmp_path, monkeypatch):
    import analysis.paper_tables as pt

    monkeypatch.setattr(pt, "ARTIFACTS", tmp_path)
    out = pt.tab2_named_methods()
    text = out.read_text()
    assert "FCD" in text and "0.675" in text
    assert text.count("dagger") >= 2  # jmi + cmim flagged as 8-dataset
    # The shipped estimator (gate + max aggregation) is withheld behind the
    # \shipname macro for double-blind review; the row is labeled by the macro,
    # and its code identifier must not leak into the generated table.
    assert text.lower().count("modmrmr") == 0
    assert r"gate (\texttt{max}) = \shipname" in text and "this work" in text


def test_leaderboard_embeds_caption_and_label(tmp_path, monkeypatch):
    import analysis.paper_tables as pt

    monkeypatch.setattr(pt, "ARTIFACTS", tmp_path)
    out = pt.tab_appendix_leaderboard()
    text = out.read_text()
    # Regeneration must be self-contained: the longtable carries its own
    # caption/label so a rebuild never orphans \ref{tab:leaderboard}.
    assert r"\label{tab:leaderboard}" in text
    assert r"\caption{" in text


def test_t1_matrix_shape(tmp_path, monkeypatch):
    import analysis.paper_tables as pt

    monkeypatch.setattr(pt, "ARTIFACTS", tmp_path)
    out = pt.tab1_measure_dependence()
    text = out.read_text()
    for rel in ["pearson", "spearman", "mutual", "distance"]:
        assert rel in text


def test_leaderboard_has_180_specs(tmp_path, monkeypatch):
    import analysis.paper_tables as pt

    monkeypatch.setattr(pt, "ARTIFACTS", tmp_path)
    prep = pt.prep_leaderboard()
    assert len(prep) == 180
    assert prep["f1"].is_monotonic_decreasing


def test_t3_dataset_inventory_covers_all_19(tmp_path, monkeypatch):
    import analysis.paper_tables as pt

    monkeypatch.setattr(pt, "ARTIFACTS", tmp_path)
    prep = pt.prep_datasets()
    assert len(prep) == 19
