"""LaTeX table generators for the paper (contributions C1 and C2).

Emit self-contained ``tabular`` environments as strings so they can be
``\\input``-ed by the manuscript or written to .tex files by reproduce.py.
"""

from __future__ import annotations

import pandas as pd

from analysis.ranks import method_rank_table


def _escape(text: str) -> str:
    # Underscores are the only LaTeX-special character that appears in the data
    # we emit (method names like ``mrmr_smazzanti``/``ModMRMR_mi``). Regime labels
    # ("p>>n"/"n>=p") are passed around _escape, not through it.
    return text.replace("_", r"\_")


def design_space_table() -> str:
    """Static C1 taxonomy: named methods as occupied operator x aggregation cells."""
    header = (
        "\\begin{tabular}{lllll}\n\\toprule\n"
        "Method & Operator & Aggregation & Relevance & Redundancy \\\\\n\\midrule\n"
    )
    rows = [
        ("MID / FCD", "difference", "mean", "MI / F-test", "MI / |corr|"),
        ("MIQ / FCQ", "quotient", "mean", "MI / F-test", "MI / |corr|"),
        ("JMI", "conditional-MI", "mean", "MI", "conditional MI"),
        ("CMIM", "conditional-MI", "max", "MI", "conditional MI"),
        (
            "\\textbf{ModMRMR}",
            "\\textbf{multiplicative}",
            "\\textbf{max}",
            "injectable",
            "injectable",
        ),
    ]
    body = "".join(" & ".join(r) + " \\\\\n" for r in rows)
    return header + body + "\\bottomrule\n\\end{tabular}"


def win_rank_table(results: pd.DataFrame, task: str) -> str:
    """Per-method mean rank and win count (datasets where the method ranks best)."""
    ranks = method_rank_table(results, task)
    per_dataset = ranks.drop(index="mean_rank")
    # A "win" is ranking best on a dataset. Under average-ranking a tie for best
    # yields rank 1.5 (not 1.0), so compare each dataset row to its own minimum
    # rather than to the literal 1.0 — otherwise tied winners score zero wins.
    wins = per_dataset.eq(per_dataset.min(axis=1), axis=0).sum(axis=0)
    mean_rank = ranks.loc["mean_rank"]
    header = "\\begin{tabular}{lrr}\n\\toprule\nMethod & Mean rank & Wins \\\\\n\\midrule\n"
    order = mean_rank.sort_values().index
    body = "".join(f"{_escape(str(m))} & {mean_rank[m]:.2f} & {int(wins[m])} \\\\\n" for m in order)
    return header + body + "\\bottomrule\n\\end{tabular}"


def decision_table(guide: dict[tuple[str, str], dict]) -> str:
    """C2 'which criterion when' table from build_decision_guide output."""
    header = (
        "\\begin{tabular}{lll}\n\\toprule\n"
        "Task & Data regime & Recommended criterion \\\\\n\\midrule\n"
    )
    body = ""
    for (task, regime), entry in sorted(guide.items()):
        runner_up = entry["ranking"][1][0] if len(entry["ranking"]) > 1 else "--"
        rec = f"{entry['recommended']} (then {runner_up})"
        # regime labels ("p>>n" / "n>=p") are kept literal (not LaTeX-escaped) —
        # they render fine in text mode and match the contract's data_regime() values.
        body += f"{task} & {regime} & {_escape(rec)} \\\\\n"
    return header + body + "\\bottomrule\n\\end{tabular}"
