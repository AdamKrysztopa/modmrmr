import pandas as pd

from analysis.guidance import build_decision_guide
from analysis.tables import decision_table, design_space_table, win_rank_table


def test_design_space_table_contains_key_cells() -> None:
    tex = design_space_table()
    assert "\\begin{tabular}" in tex and "\\end{tabular}" in tex
    for token in ["ModMRMR", "multiplicative", "max", "MIQ", "CMIM"]:
        assert token in tex


def test_win_rank_table_has_methods_and_numbers(synthetic_results: pd.DataFrame) -> None:
    tex = win_rank_table(synthetic_results, "classification")
    assert "\\begin{tabular}" in tex
    assert "ModMRMR" in tex
    assert "Mean rank" in tex


def test_decision_table_lists_recommendations(synthetic_results: pd.DataFrame) -> None:
    guide = build_decision_guide(synthetic_results)
    tex = decision_table(guide)
    assert "\\begin{tabular}" in tex
    assert "classification" in tex
    # Regime labels are emitted literally (contract's data_regime values), not as
    # $\gg$ — pin the exact form so the escaping contract can't silently drift.
    assert "p>>n" in tex


def _single_k_results(scores: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Minimal classification results (one k/learner/seed) from {(dataset,method): score}."""
    rows = [
        {
            "dataset": dataset,
            "method": method,
            "task": "classification",
            "metric": "balanced_accuracy",
            "k": 5,
            "learner": "knn",
            "seed": 0,
            "score": score,
        }
        for (dataset, method), score in scores.items()
    ]
    return pd.DataFrame(rows)


def test_win_rank_table_escapes_underscored_method_names_and_counts_tied_wins() -> None:
    # Two methods tie for best on every dataset; average-ranking gives them rank
    # 1.5, so both must count as winners (not zero), and the underscored real
    # method names must be LaTeX-escaped.
    results = _single_k_results(
        {
            ("d1", "mrmr_smazzanti"): 0.9,
            ("d1", "ModMRMR_mi"): 0.9,
            ("d1", "skb_f"): 0.5,
            ("d2", "mrmr_smazzanti"): 0.9,
            ("d2", "ModMRMR_mi"): 0.9,
            ("d2", "skb_f"): 0.5,
        }
    )
    tex = win_rank_table(results, "classification")
    assert r"mrmr\_smazzanti" in tex
    assert r"ModMRMR\_mi" in tex
    assert "_" not in tex.replace(r"\_", "")  # no unescaped underscore survives
    # Each tied winner wins both datasets; the loser wins none.
    assert r"mrmr\_smazzanti & 1.50 & 2" in tex
    assert r"ModMRMR\_mi & 1.50 & 2" in tex
    assert r"skb\_f & 3.00 & 0" in tex
