"""Paired significance tests for the real high-dimensional downstream arm (F2.3, H-5).

Reads ``results/highdim_study.parquet`` (kind == "real"; 4 OpenML high-dim sets x
10 seeds x {difference, multiplicative|mean (the gate), quotient|mean,
multiplicative|max} x k in {5, 10, 20}) and runs a paired Wilcoxon signed-rank
test, seed-matched, of the gate (``multiplicative``+``mean``) against the
quotient and against ``difference``, per dataset and pooled across the 4
datasets. Writes ``results/stats_highdim_real_paired.csv``.

The regularized-quotient arm (``reg_quotient``) is a synthetic-only arm (see
``analysis/highdim_regquotient.py``) and was never run on the real datasets, so
no gate-vs-reg_quotient row is emitted here; this is noted in the run log.

Emits incremental progress even though this is a fast, non-checkpointed step: a
long silent run is indistinguishable from a hang.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GATE = ("multiplicative", "mean")
COMPARISONS = {
    "gate_vs_quotient": (GATE, ("quotient", "mean")),
    "gate_vs_difference": (GATE, ("difference", "mean")),
}
REAL_DATASETS = ["madelon", "isolet", "riboflavin", "arcene"]


def log(msg: str) -> None:
    print(msg, flush=True)


def _pivot(real: pd.DataFrame, op_agg: tuple[str, str], k: int, datasets: list[str]) -> pd.Series:
    """Downstream score indexed by (dataset, seed) for one (operator, aggregation, k) cell."""
    op, agg = op_agg
    sub = real[
        (real.operator == op)
        & (real.aggregation == agg)
        & (real.k == k)
        & (real.dataset.isin(datasets))
    ]
    return sub.set_index(["dataset", "seed"])["downstream_score"].sort_index()


def _paired_test(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Wilcoxon signed-rank on the seed-matched pair; returns (mean_diff, p, n)."""
    idx = a.index.intersection(b.index)
    av, bv = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    diff = av - bv
    if np.allclose(diff, 0.0):
        return float(diff.mean()), 1.0, len(idx)
    stat_res = wilcoxon(av, bv, zero_method="wilcox", alternative="two-sided")
    return float(diff.mean()), float(stat_res.pvalue), len(idx)


def run(parquet_path: Path, out_path: Path, ks: list[int], datasets: list[str]) -> pd.DataFrame:
    log(f"reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    real = df[df.kind == "real"].copy()
    missing = set(datasets) - set(real.dataset.unique())
    if missing:
        log(f"WARNING: requested datasets not present in real arm: {sorted(missing)}")
    datasets = [d for d in datasets if d in set(real.dataset.unique())]
    seeds = sorted(real.seed.unique())
    log(f"real arm: datasets={datasets} seeds={seeds} (n={len(seeds)}) ks={ks}")

    rows: list[dict] = []
    total = len(COMPARISONS) * len(ks) * (len(datasets) + 1)
    done = 0
    for name, (a_spec, b_spec) in COMPARISONS.items():
        for k in ks:
            a_all = _pivot(real, a_spec, k, datasets)
            b_all = _pivot(real, b_spec, k, datasets)
            for ds in datasets:
                a_ds = a_all.xs(ds, level="dataset", drop_level=False)
                b_ds = b_all.xs(ds, level="dataset", drop_level=False)
                mean_diff, p, n = _paired_test(a_ds, b_ds)
                rows.append(
                    {
                        "comparison": name,
                        "scope": ds,
                        "k": k,
                        "n_seeds": n,
                        "mean_a": float(a_ds.mean()),
                        "mean_b": float(b_ds.mean()),
                        "mean_diff": mean_diff,
                        "wilcoxon_p": p,
                    }
                )
                done += 1
                log(
                    f"  [{done}/{total}] {name} {ds:11s} k={k:>2} "
                    f"a={a_ds.mean():.3f} b={b_ds.mean():.3f} p={p:.4f}"
                )
            mean_diff, p, n = _paired_test(a_all, b_all)
            rows.append(
                {
                    "comparison": name,
                    "scope": "pooled",
                    "k": k,
                    "n_seeds": n,
                    "mean_a": float(a_all.mean()),
                    "mean_b": float(b_all.mean()),
                    "mean_diff": mean_diff,
                    "wilcoxon_p": p,
                }
            )
            done += 1
            log(
                f"  [{done}/{total}] {name} {'pooled':11s} k={k:>2} "
                f"a={a_all.mean():.3f} b={b_all.mean():.3f} p={p:.4f} (n={n} dataset x seed pairs)"
            )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "scope",
            "k",
            "n_seeds",
            "mean_a",
            "mean_b",
            "mean_diff",
            "wilcoxon_p",
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log(f"wrote {len(out_df)} rows -> {out_path}")
    log(
        "note: reg_quotient is a synthetic-only arm (analysis/highdim_regquotient.py); it was "
        "never run on the real high-dim datasets, so no gate_vs_reg_quotient row exists here."
    )
    return out_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.highdim_real_paired_stats", description=__doc__)
    parser.add_argument("--parquet", type=str, default="results/highdim_study.parquet")
    parser.add_argument("--out", type=str, default="results/stats_highdim_real_paired.csv")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--datasets", nargs="+", default=REAL_DATASETS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(Path(args.parquet), Path(args.out), args.ks, list(args.datasets))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
