"""Variance-based effect sizes for the design axes (certifies the Rule-1 headline).

The paper's headline is that the relevance measure dominates the operator, and
does so *as an interaction with dependence structure*. A raw within-nonlinear
F1 span (0.285) over a pooled operator span (0.041) gives the "~7x" figure, but
that compares a conditional span to a marginal one. This script computes an
honest, variance-based decomposition on the *existing* golden factorial
(no new runs): unbiased omega-squared per factor from an ANOVA of recovery F1.

Reads ``results/factorial.parquet`` (golden rows: ``f1`` non-null, no error),
collapses to one mean F1 per ``dataset x relevance x redundancy x operator x
aggregation`` cell (averaging out seed / stopping / k, which are operating-point
nuisance factors), and fits two OLS models with Type II sums of squares:

  A. main effects only (dataset + relevance + redundancy + operator + aggregation)
  B. relevance x dependence interaction (+ redundancy + operator + aggregation)

Type II SS is used because the design is unbalanced/incomplete: task-specific
relevance measures (``f_classif``/``mutual_info_classif`` on classification,
``f_regression``/``mutual_info_regression`` on regression) mean relevance x
dataset is not fully crossed. Writes ``results/stats_effect_sizes.csv``.

Emits incremental progress even though this is a fast, non-checkpointed step: a
long silent run is indistinguishable from a hang.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FACTORIAL_PARQUET = Path("results/factorial.parquet")
OUT_CSV = Path("results/stats_effect_sizes.csv")

MODELS = {
    "main_effects": (
        "f1 ~ C(relevance) + C(redundancy) + C(operator) + C(aggregation) + C(dataset)"
    ),
    "relevance_x_dependence": (
        "f1 ~ C(relevance) * C(dependence) + C(redundancy) + C(operator) + C(aggregation)"
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _omega_squared(formula: str, cell: pd.DataFrame) -> pd.DataFrame:
    """Unbiased omega-squared and eta-squared per term from a Type II ANOVA."""
    model = smf.ols(formula, data=cell).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ms_resid = aov.loc["Residual", "sum_sq"] / aov.loc["Residual", "df"]
    ss_total = aov["sum_sq"].sum()
    rows = []
    for term in aov.index:
        if term == "Residual":
            continue
        ss, df = aov.loc[term, "sum_sq"], aov.loc[term, "df"]
        rows.append(
            {
                "term": term,
                "df": int(df),
                "omega2": (ss - df * ms_resid) / (ss_total + ms_resid),
                "eta2": ss / ss_total,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=FACTORIAL_PARQUET)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    log(f"[effect_sizes] reading {args.parquet}")
    df = pd.read_parquet(args.parquet)
    golden = df[df["f1"].notna() & df["error"].isna()]
    log(f"[effect_sizes] {len(golden):,} golden rows over {golden['dataset'].nunique()} datasets")

    keys = ["dataset", "dependence", "relevance", "redundancy", "operator", "aggregation"]
    cell = golden.groupby(keys, as_index=False)["f1"].mean()
    log(f"[effect_sizes] collapsed to {len(cell):,} cell means")

    frames = []
    for name, formula in MODELS.items():
        log(f"[effect_sizes] fitting model '{name}': {formula}")
        out = _omega_squared(formula, cell)
        out.insert(0, "model", name)
        frames.append(out)
        for _, r in out.iterrows():
            log(f"    {r['term']:34s} omega2={r['omega2']:.4f}  eta2={r['eta2']:.4f}")

    result = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    log(f"[effect_sizes] wrote {args.out}")


if __name__ == "__main__":
    main()
