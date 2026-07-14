"""LaTeX tables for the OpenMRMR paper: T1 measure x dependence, T2 named methods
vs external baselines, T3 dataset inventory, appendix 180-spec leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.display_names import display_spec
from analysis.paper_figures import FACTORIAL, RELEVANCE_ORDER, golden, prep_measure_dependence
from analysis.paths import ARTIFACTS
from benchmarks.datasets import DATASETS as BENCHMARK_DATASETS
from mechanism.datasets import MECHANISM_DATASETS, load_mechanism_dataset

BASELINES_CSV = Path("results/baseline_comparison_summary.csv")
LEADERBOARD_CSV = Path("results/factorial_leaderboard.csv")

DEPENDENCE_COLS = ["linear", "nonlinear", "mixed"]


def _escape(value: object) -> object:
    """Escape LaTeX-hostile underscores in string cells; pass everything else through."""
    if isinstance(value, str):
        return value.replace("_", r"\_")
    return value


def to_booktabs(
    df: pd.DataFrame,
    path: Path,
    *,
    float_fmt: str = "%.3f",
    bold_max_cols: tuple[str, ...] = (),
    longtable: bool = False,
    caption: str | None = None,
    label: str | None = None,
    resizebox: bool = False,
    font: str | None = None,
    sideways: bool = False,
    tabcolsep: str | None = None,
) -> Path:
    """Write ``df`` as a standalone booktabs (or longtable) LaTeX fragment.

    Float columns are formatted with ``float_fmt``; the max value in any column
    named in ``bold_max_cols`` is wrapped in ``\\textbf{}``. Underscores in string
    (object-dtype) cells and column headers are escaped. Always emits ``index=False``
    — callers put every displayed field in a real column beforehand.

    Width control, for tables that would otherwise run past the text block:

    * ``font`` wraps the ``tabular`` in a size group (e.g. ``"scriptsize"``).
    * ``tabcolsep`` shrinks the inter-column padding (e.g. ``"3.5pt"``). With many
      columns this is often the cheapest win: padding is charged twice per column.
    * ``sideways`` emits a ``sidewaystable`` (rotating package) on its own page,
      which is the only real fix for a table that is wide in *content* rather than
      merely in padding.

    ``resizebox`` is retained for callers outside this class, but note that
    ``sn-jnl.cls`` patches ``tabular`` so it cannot be typeset inside a box
    argument — ``\\resizebox{...}{!}{\\begin{tabular}...}`` is a hard error under
    the Springer class. Use ``font``/``tabcolsep``/``sideways`` instead.

    These are emitted by the generator rather than hand-edited into the artifact:
    the artifacts are overwritten on every ``paper_tables`` run, so a manual
    ``\\scriptsize`` added to the ``.tex`` silently disappears the next time this
    script runs and the table quietly overflows the margin again.
    """
    out = df.reset_index(drop=True).copy()

    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            if col in bold_max_cols:
                best = out[col].max()
                out[col] = out[col].apply(
                    lambda v, best=best: (
                        f"\\textbf{{{float_fmt % v}}}" if v == best else float_fmt % v
                    )
                )
            else:
                out[col] = out[col].apply(lambda v: float_fmt % v)
        elif not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].apply(_escape)

    out.columns = [_escape(c) for c in out.columns]

    path.parent.mkdir(parents=True, exist_ok=True)
    latex = out.to_latex(
        index=False, escape=False, longtable=longtable, caption=caption, label=label
    )
    if resizebox and not longtable:
        latex = latex.replace(
            "\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1
        )
        latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)

    if not longtable:
        preamble = ""
        if font:
            preamble += f"{{\\{font}\n"
        if tabcolsep:
            preamble += f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}%\n"
        if preamble:
            latex = latex.replace("\\begin{tabular}", preamble + "\\begin{tabular}", 1)
            if font:
                latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
        if sideways:
            latex = latex.replace("\\begin{table}", "\\begin{sidewaystable}", 1)
            latex = latex.replace("\\end{table}", "\\end{sidewaystable}", 1)

    path.write_text(latex)
    print(path, flush=True)
    return path


def tab1_measure_dependence() -> Path:
    """T1: 7 relevance rows x 3 dependence columns, mean recovery F1, best bolded."""
    df = pd.read_parquet(FACTORIAL)
    prep = prep_measure_dependence(df)
    pivot = (
        prep.pivot(index="relevance", columns="dependence", values="mean")
        .reindex(index=RELEVANCE_ORDER, columns=DEPENDENCE_COLS)
        .reset_index()
    )
    return to_booktabs(
        pivot,
        ARTIFACTS / "tab1_measure_dependence.tex",
        bold_max_cols=tuple(DEPENDENCE_COLS),
    )


def tab2_named_methods() -> Path:
    """T2: named design-space methods vs external baselines, from baseline_comparison_summary.csv.

    Rows evaluated only on the 8 classification-only golden datasets (jmi, cmim)
    get a dagger footnote marker on the method name. The ModMRMR row is relabeled
    to the generic operator name: the brand appears only in the implementation
    section (design-spec locked decision).
    """
    df = pd.read_csv(BASELINES_CSV).sort_values("mean_recovery_f1", ascending=False)
    df["method"] = df["method"].replace({"ModMRMR": r"gate (\texttt{max}) = \shipname"})
    df["source"] = df["source"].replace({"modmrmr": "this work"})
    classification_only = df["n_datasets"] < df["n_datasets"].max()
    method = df["method"].where(~classification_only, df["method"] + r"\textsuperscript{$\dagger$}")
    out = pd.DataFrame(
        {
            "method": method,
            "source": df["source"],
            "mean F1": df["mean_recovery_f1"],
            "mean noise rate": df["mean_noise_rate"],
            "\\# datasets": df["n_datasets"],
        }
    )
    path = to_booktabs(out, ARTIFACTS / "tab2_named_methods.tex")
    footnote = (
        "\n% \\textsuperscript{$\\dagger$}: evaluated only on the 8 classification-only\n"
        "% golden datasets (jmi, cmim require discretized inputs).\n"
    )
    path.write_text(path.read_text() + footnote)
    return path


def prep_datasets() -> pd.DataFrame:
    """19-row dataset inventory: name, task, n, p, dependence, #informative, #redundant.

    n/p/ground-truth counts for the 15 mechanism (golden) datasets come from loading
    each once via the ``mechanism.datasets`` registry; the 4 benchmark-only sets
    (breast_cancer, diabetes, friedman1, synthetic_clf) have no GroundTruth object, so
    their n/p come from the ``benchmarks.datasets`` registry metadata and the
    informative/redundant cells are em-dashes.
    """
    factorial_meta = pd.read_parquet(FACTORIAL).groupby("dataset")[["dependence", "task"]].first()
    rows = []
    for name in sorted(factorial_meta.index):
        dependence = factorial_meta.loc[name, "dependence"]
        task = factorial_meta.loc[name, "task"]
        if name in MECHANISM_DATASETS:
            X, _y, _task, gt = load_mechanism_dataset(name)
            n, p = X.shape
            n_informative = str(len(gt.informative))
            n_redundant = str(sum(len(group) for group in gt.codependent))
        else:
            entry = BENCHMARK_DATASETS[name]
            n, p = entry["n"], entry["p"]
            n_informative = "---"
            n_redundant = "---"
        rows.append(
            {
                "dataset": name,
                "task": task,
                "n": n,
                "p": p,
                "dependence": dependence,
                "\\# informative": n_informative,
                "\\# redundant": n_redundant,
            }
        )
    return pd.DataFrame(rows)


def tab3_datasets() -> Path:
    """T3: dataset inventory (19 rows)."""
    prep = prep_datasets()
    # Seven columns, several holding long names: the inter-column padding alone
    # is charged 14 times, so tightening it (plus a size drop) is what keeps this
    # inside the text block.
    return to_booktabs(
        prep,
        ARTIFACTS / "tab3_datasets.tex",
        font="scriptsize",
        tabcolsep="3.5pt",
    )


def prep_leaderboard() -> pd.DataFrame:
    """All 180 specs, mean f1 / noise_rate / average_precision over golden sets,
    ranked by f1 descending with a 1-based ``rank`` column."""
    df = golden(pd.read_parquet(FACTORIAL))
    per_spec = (
        df.groupby("spec", as_index=False)[["f1", "noise_rate", "average_precision"]]
        .mean()
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    per_spec.insert(0, "rank", per_spec.index + 1)
    return per_spec


# Classical named methods, keyed to their exact spec label
# (relevance-short | redundancy | operator | aggregation). ModMRMR is the
# multiplicative gate at max aggregation; its headline spec is the best-scoring
# mult|max cell, resolved from the data rather than hard-coded.
NAMED_SPECS = {
    "mi|mutual_info_sklearn|difference|mean": "MID",
    "mi|mutual_info_sklearn|quotient|mean": "MIQ",
    "f|pearson_abs|difference|mean": "FCD",
    "f|pearson_abs|quotient|mean": "FCQ",
}


def write_leaderboard_csv() -> Path:
    """Write the full 180-row leaderboard to ``results/factorial_leaderboard.csv``,
    the released artifact referenced by the appendix table."""
    LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    prep_leaderboard().to_csv(LEADERBOARD_CSV, index=False)
    print(LEADERBOARD_CSV, flush=True)
    return LEADERBOARD_CSV


def prep_leaderboard_slice() -> pd.DataFrame:
    """Curated ~15-row slice of the 180-spec leaderboard: top 10 by F1, the named
    classical methods (MID/MIQ/FCD/FCQ), the recommended gate, the shipped
    \\texttt{ModMRMR} estimator, and the bottom 3.

    The ``rank`` column (out of 180) exposes the gaps, so the reader sees this is a
    slice, not the full board. The recommended-gate label is attached to the
    best-scoring ``multiplicative|mean`` spec (the gate, mean aggregation); the
    ModMRMR label is attached to the best-scoring ``multiplicative|max`` spec (the
    gate, max aggregation) — both resolved from the data rather than hard-coded."""
    full = prep_leaderboard()
    named = dict(NAMED_SPECS)
    mult_mean = full[full["spec"].str.contains(r"multiplicative\|mean", regex=True)]
    if not mult_mean.empty:
        named[mult_mean.iloc[0]["spec"]] = "gate (recommended)"
    mult_max = full[full["spec"].str.contains(r"multiplicative\|max", regex=True)]
    if not mult_max.empty:
        named[mult_max.iloc[0]["spec"]] = r"gate (\texttt{max}) = \shipname"

    top = full.head(10).copy()
    bottom = full.tail(3).copy()
    named_rows = full[full["spec"].isin(named)].copy()
    slice_df = pd.concat([top, named_rows, bottom]).drop_duplicates("spec")
    slice_df = slice_df.sort_values("rank").reset_index(drop=True)
    slice_df["method"] = slice_df["spec"].map(named).fillna("")
    return slice_df


def tab_appendix_leaderboard() -> Path:
    """Appendix: a curated slice of the 180-spec leaderboard (reviewer B-14) —
    replaces the former 4-page longtable. Top 10 by mean golden-set recovery F1,
    the named classical methods, the recommended gate, the shipped ModMRMR
    estimator, and the bottom 3, with rank out of 180.

    The ``spec`` column is rendered through :func:`display_spec` so the operator
    token ``multiplicative`` reads as "gate" (the display-name decision of
    fix-plan-v3 F1.1-display); ``results/factorial_leaderboard.csv``, the
    released CSV this table is sliced from, keeps the raw ``multiplicative``
    token — the caption explains the mapping once.
    """
    slice_df = prep_leaderboard_slice()
    out = slice_df[["rank", "spec", "method", "f1", "noise_rate", "average_precision"]].rename(
        columns={
            "spec": "spec",
            "method": "method",
            "f1": "mean f1",
            "noise_rate": "mean noise rate",
            "average_precision": "mean AP",
        }
    )
    out["spec"] = out["spec"].map(display_spec)
    return to_booktabs(
        out,
        ARTIFACTS / "tab_appendix_leaderboard.tex",
        caption=(
            "Curated slice of the 180-specification golden-set leaderboard (10 "
            "seeds), ranked by mean recovery F1, the study's primary recovery "
            "metric (the AP ordering differs; Sec.~\\ref{sec:discussion}): the top "
            "10, the named classical methods (MID, MIQ, FCD, FCQ), \\shipname "
            "(the gate with \\texttt{max} "
            "aggregation), and the bottom 3. The rank column (out of 180) shows "
            "this is a slice; the full board ships as "
            "\\texttt{results/factorial\\_leaderboard.csv}, where the operator "
            "token \\texttt{multiplicative} denotes the gate, Eq.~\\eqref{eq:gate}."
        ),
        label="tab:leaderboard",
        # Landscape, on its own page: this table is wide in *content* (long
        # pipe-separated spec strings), not merely in padding, so shrinking the
        # font alone still ran ~133pt past the text block. `resizebox` is not an
        # option under sn-jnl.cls -- see to_booktabs.
        sideways=True,
        # Rotated, the table has width to spare, so it does not need to be as
        # small as it did when it was fighting the portrait text block.
        font="footnotesize",
    )


FACTORIAL_SUMMARY_CSV = Path("results/factorial_summary.csv")
HIGHDIM_SUMMARY_CSV = Path("results/highdim_study_summary.csv")

# The three named mRMR-family operators, in display order, with their labels.
_PR_OPERATORS = (
    ("difference", "MID (difference)"),
    ("quotient", "MIQ (quotient)"),
    ("multiplicative", "Gate (this work)"),
)
# Matched setting for the low-dimension panel (the boxplot's best measure pairing).
_PR_MATCHED = {"rel": "dcor", "red": "pearson_abs", "agg": "mean"}
_PR_HIGHDIM_P = 10_000
_PR_HIGHDIM_K = 20  # fixed selection budget for the high-dim panel
_N_INFORMATIVE = 5  # golden/high-dim synthetic families plant 5 informative features


def _harmonic_f1(precision: pd.Series, recall: pd.Series) -> pd.Series:
    """F1 as the harmonic mean of precision and recall, 0 where both are 0."""
    denom = precision + recall
    return (2 * precision * recall / denom).where(denom > 0, 0.0)


def _pr_low_dim() -> pd.DataFrame:
    """Matched-setting precision/recall/F1/noise per operator on the golden suite."""
    df = pd.read_csv(FACTORIAL_SUMMARY_CSV)
    parts = df["spec"].astype(str).str.split("|", expand=True)
    for i, name in enumerate(("rel", "red", "op", "agg")):
        df[name] = parts[i].astype(str).str.strip()
    matched = df[
        (df["rel"] == _PR_MATCHED["rel"])
        & (df["red"] == _PR_MATCHED["red"])
        & (df["agg"] == _PR_MATCHED["agg"])
    ]
    panel = matched.groupby("op")[["precision", "recall", "noise_rate"]].mean()
    # F1 is the harmonic mean of the *tabulated* precision and recall so every
    # row is internally consistent (a stored mean-of-per-run F1 would not
    # reconcile with the P and R shown alongside it).
    panel["f1"] = _harmonic_f1(panel["precision"], panel["recall"])
    return panel


def _pr_high_dim() -> pd.DataFrame:
    """High-dimension (p=10,000, mean agg) precision/recall/F1/noise per operator.

    The synthetic sweep is fixed-budget (k features over 5 informative), so
    precision = informative hits / k = recall * 5 / k — derived here so the panel
    reconciles exactly with the recall of the quotient-collapse figure rather than
    being a second, separately-scored measurement. F1 is then the harmonic mean of
    that derived precision and the recall (the stored ``recovery_f1`` column is a
    contamination score equal to 1 - noise_rate, not an F1, so it is not used)."""
    df = pd.read_csv(HIGHDIM_SUMMARY_CSV)
    for col in ("kind", "operator", "aggregation"):
        df[col] = df[col].astype(str)
    sl = df[
        (df["kind"] == "synthetic")
        & (df["aggregation"] == "mean")
        & (df["p"] == _PR_HIGHDIM_P)
        & (df["k"] == _PR_HIGHDIM_K)
    ].copy()
    sl["precision"] = sl["recall_informative"] * _N_INFORMATIVE / sl["k"]
    out = (
        sl.groupby("operator")[["precision", "recall_informative", "noise_rate"]]
        .mean()
        .rename(columns={"recall_informative": "recall"})
    )
    out["f1"] = _harmonic_f1(out["precision"], out["recall"])
    return out


def _pr_rows(panel: pd.DataFrame) -> list[str]:
    """Format one panel's operator rows, bolding the best value per column
    (max for precision/recall/F1, min for noise rate; rounded ties both bold)."""
    cols = ["precision", "recall", "f1", "noise_rate"]
    best = {c: (panel[c].min() if c == "noise_rate" else panel[c].max()) for c in cols}
    best_str = {c: f"{best[c]:.2f}" for c in cols}
    lines = []
    for op, label in _PR_OPERATORS:
        cells = []
        for c in cols:
            s = f"{panel.loc[op, c]:.2f}"
            cells.append(f"\\textbf{{{s}}}" if s == best_str[c] else s)
        lines.append(f"{label}  & " + " & ".join(cells) + r" \\")
    return lines


def tab_operator_pr() -> Path:
    """Straightforward precision/recall head-to-head of the three named operators
    (MID/MIQ/gate) at matched settings, in two regimes — the table a practitioner
    asks for. Reproduced from ``factorial_summary.csv`` (low dim) and
    ``highdim_study_summary.csv`` (p=10,000)."""
    low = _pr_rows(_pr_low_dim())
    high = _pr_rows(_pr_high_dim())
    caption = (
        "The three named mRMR-family operators head-to-head on plain "
        "\\emph{precision} and \\emph{recall}, at matched settings "
        "(distance-correlation relevance, $|$Pearson$|$ redundancy, mean "
        "aggregation). Precision is the fraction of selected features that are "
        "informative, recall the fraction of informative features recovered; "
        "precision is \\emph{not} $1 - \\text{noise rate}$, as the ground truth "
        "also has a redundant bucket, and F1 here is the harmonic mean of the "
        "tabulated precision and recall. Bold marks the best value per column "
        "within each regime. (High-dimension precision is capped near $5/k = 0.25$ "
        "by the fixed budget $k = 20$ over 5 informative features, so it reconciles "
        "with the recall of Fig.~\\ref{fig:quotient-collapse}.)"
    )
    body = "\n".join(
        [
            r"\begin{table}[t]\centering\small",
            f"\\caption{{{caption}}}",
            r"\label{tab:operator-pr}",
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            r"Operator & Precision & Recall & F1 & Noise rate \\",
            r"\midrule",
            r"\multicolumn{5}{@{}l}{\emph{Typical dimension} --- golden suite, "
            r"15 datasets $\times$ 10 seeds} \\",
            *low,
            r"\midrule",
            r"\multicolumn{5}{@{}l}{\emph{High dimension} --- $p = 10{,}000$ "
            r"synthetic sweep, $k = 20$, 10 seeds} \\",
            *high,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path = ARTIFACTS / "tab_operator_pr.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    print(path, flush=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.paper_tables", description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ARTIFACTS,
        help=f"output directory for table .tex fragments (default: {ARTIFACTS})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    # Module-level-default reassignment (same rationale as analysis.paper_figures):
    # every tab* builder and to_booktabs() call below reads the module global
    # ARTIFACTS directly rather than taking a path argument, so reassigning the
    # global here before dispatch resolves --outdir for all of them at once.
    global ARTIFACTS
    args = build_parser().parse_args(argv)
    ARTIFACTS = args.outdir
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for builder in (
        tab1_measure_dependence,
        tab2_named_methods,
        tab3_datasets,
        tab_appendix_leaderboard,
        tab_operator_pr,
    ):
        builder()
    write_leaderboard_csv()


if __name__ == "__main__":
    main()
