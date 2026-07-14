"""Equivalence / non-inferiority of the operators on recovery F1, and the redundancy axis.

The paper's operator recommendation rests on a two-part claim: the quotient fails at
high dimension (Sec. 5.2), and switching away from it costs nothing at typical
dimension (Sec. 5.1). The second half was previously warranted by a *non-significant*
omnibus Friedman test (p = 0.549) -- i.e. by accepting a null. A failure to reject is
not evidence of equivalence, and this script replaces that inference with two
quantities a reviewer can check:

  1. The equivalence margin the design can actually support. The 90% CI on the paired
     per-block difference is the TOST interval at alpha = 0.05: equivalence holds only
     for margins wider than its larger absolute bound. Reporting that number is what
     makes "we cannot establish equivalence" a measurement rather than a hedge.

  2. The targeted directional contrast. The Friedman omnibus is a three-way rank test
     that the difference operator's tie with the gate washes out; the gate-vs-quotient
     paired contrast is the comparison the recommendation actually needs, and it is
     one-sided in the gate's favour.

Both are reported under BOTH blockings -- 15 datasets, and 11 mechanisms after
collapsing the clf/reg twins (the pseudo-replication guard of
``analysis/hardening_outputs.py``). The mechanism blocking is the conservative primary;
the dataset blocking is the sensitivity check.

The same script also settles the redundancy axis, which the factorial declares but which
had no reported result: it is third-order on recovery F1 and non-null on noise rate.

Reads ``results/factorial.parquet``; writes ``results/stats_operator_equivalence.csv``
and ``results/stats_redundancy_axis.csv``. No new runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon
from scipy.stats import t as tdist

FACTORIAL_PARQUET = Path("results/factorial.parquet")
OUT_EQUIV = Path("results/stats_operator_equivalence.csv")
OUT_REDUNDANCY = Path("results/stats_redundancy_axis.csv")

GATE = "multiplicative"
BLOCKINGS = ("mechanism", "dataset")  # primary first
CONTRASTS = ((GATE, "quotient"), (GATE, "difference"), ("difference", "quotient"))


def log(msg: str) -> None:
    print(msg, flush=True)


def _golden(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    golden = df[df["f1"].notna() & df["error"].isna()].copy()
    golden["mechanism"] = golden["dataset"].str.replace(r"_(clf|reg)$", "", regex=True)
    return golden


def _paired_contrast(a: pd.Series, b: pd.Series, *, alpha: float = 0.05) -> dict:
    """Paired difference a - b with the TOST interval and the directional tests.

    The 90% CI (for alpha = 0.05) is the two-one-sided-tests interval: equivalence at
    margin delta holds iff the interval lies inside (-delta, +delta), so the larger
    absolute bound is the *smallest* margin at which this design could conclude
    equivalence. That bound is the honest summary of what the study can rule out.
    """
    diff = (a - b).to_numpy()
    n = len(diff)
    mean, sd = float(diff.mean()), float(diff.std(ddof=1))
    se = sd / np.sqrt(n)

    half_90 = float(tdist.ppf(1 - alpha, n - 1)) * se
    half_95 = float(tdist.ppf(1 - alpha / 2, n - 1)) * se
    lo90, hi90 = mean - half_90, mean + half_90

    t_stat, t_p = ttest_rel(a.to_numpy(), b.to_numpy())
    w_stat, w_p = wilcoxon(a.to_numpy(), b.to_numpy())

    return {
        "n_blocks": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "ci95_lo": mean - half_95,
        "ci95_hi": mean + half_95,
        "tost90_lo": lo90,
        "tost90_hi": hi90,
        # smallest equivalence margin this design can support, at alpha
        "equiv_margin_min": max(abs(lo90), abs(hi90)),
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "cohen_dz": mean / sd if sd else float("nan"),
        # one-sided at alpha: is `a` superior to `b`?
        "a_superior_at_alpha": bool(lo90 > 0),
        "a_wins_blocks": int((a > b).sum()),
    }


def _operator_equivalence(golden: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block in BLOCKINGS:
        per = golden.groupby([block, "operator"], as_index=False)["f1"].mean()
        piv = per.pivot(index=block, columns="operator", values="f1")

        stat, p = friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        log(
            f"[equivalence] block={block:9s} n={len(piv):2d}  omnibus Friedman on F1: "
            f"stat={stat:.3f} p={p:.4f}"
        )

        for a, b in CONTRASTS:
            res = _paired_contrast(piv[a], piv[b])
            rows.append(
                {"blocking": block, "contrast": f"{a}-{b}", "omnibus_friedman_p": float(p), **res}
            )
            verdict = "SUPERIOR" if res["a_superior_at_alpha"] else "not superior"
            log(
                f"    {a:14s} - {b:14s}  diff={res['mean_diff']:+.4f}  "
                f"TOST90=[{res['tost90_lo']:+.4f},{res['tost90_hi']:+.4f}]  "
                f"wilcoxon p={res['wilcoxon_p']:.4f}  dz={res['cohen_dz']:+.3f}  -> {verdict}"
            )
            log(
                f"        equivalence supportable only for margins > "
                f"{res['equiv_margin_min']:.4f} F1"
            )
    return pd.DataFrame(rows)


def _redundancy_axis(golden: pd.DataFrame) -> pd.DataFrame:
    """The declared-but-unreported axis: third-order on recovery, non-null on noise."""
    rows = []
    for metric, higher_is_better in (("f1", True), ("noise_rate", False)):
        for block in BLOCKINGS:
            per = golden.groupby([block, "redundancy"], as_index=False)[metric].mean()
            piv = per.pivot(index=block, columns="redundancy", values=metric)
            stat, p = friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
            means = piv.mean()
            span = float(means.max() - means.min())
            best = means.idxmin() if not higher_is_better else means.idxmax()

            log(
                f"[redundancy] {metric:10s} block={block:9s} n={len(piv):2d}  "
                f"Friedman stat={stat:6.3f} p={p:.6f}  span={span:.4f}  best={best}"
            )
            for name, value in means.items():
                rows.append(
                    {
                        "metric": metric,
                        "blocking": block,
                        "redundancy": str(name),
                        "mean": float(value),
                        "friedman_stat": float(stat),
                        "friedman_p": float(p),
                        "span": span,
                        "n_blocks": len(piv),
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=FACTORIAL_PARQUET)
    parser.add_argument("--out-equivalence", type=Path, default=OUT_EQUIV)
    parser.add_argument("--out-redundancy", type=Path, default=OUT_REDUNDANCY)
    args = parser.parse_args(argv)

    log(f"[equivalence] reading {args.parquet}")
    golden = _golden(args.parquet)
    log(
        f"[equivalence] {len(golden):,} golden rows | "
        f"{golden['dataset'].nunique()} datasets | {golden['mechanism'].nunique()} mechanisms"
    )

    equiv = _operator_equivalence(golden)
    redundancy = _redundancy_axis(golden)

    for frame, path in ((equiv, args.out_equivalence), (redundancy, args.out_redundancy)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        log(f"[equivalence] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
