"""Regularized-quotient arm of the high-dimensional operator study (paper 5.2).

The main :mod:`analysis.highdim_study` sweep checkpoints per ``(dataset, seed)``
shard with the four original operator variants baked into each shard file, so a
new operator cannot be added by rerunning it (a cached shard resumes without the
new column). This driver runs the single ``reg_quotient`` variant
``score = rel / (agg_red + eps)`` across the same synthetic p-sweep, into its own
checkpoint directory, so the arm can be merged with the 10-seed baseline for the
money figure. It answers reviewer A-M8: does additively regularizing the
quotient's denominator rescue it from high-dimensional collapse?

Verbose and resumable: one flushed progress line per shard, each shard
checkpointed so an interruption is recoverable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from analysis.highdim_study import (
    HIGHDIM_COLUMNS,
    _fit_selector,
    _load_checkpoint,
    _write_checkpoint,
    make_highdim_synthetic,
    recovery_metrics,
)
from mechanism.factorial import SelectorSpec
from mechanism.recovery import ranking_scores, recovery
from modmrmr.core.scorers.base import fast_pearson_penalty

# The reg_quotient arm, matched to the quotient/mean baseline in every axis but
# the operator so the comparison isolates the epsilon regularization.
REGQ_SPEC = SelectorSpec("pearson", "pearson_abs", "reg_quotient", "mean")


def _regq_shard(
    p: int, seed: int, ks: list[int], checkpoint_dir: Path, done: int, total: int
) -> pd.DataFrame:
    label = f"synthetic_p{p}"
    path = checkpoint_dir / f"{label}__{seed}.parquet"
    t0 = time.perf_counter()
    cached = _load_checkpoint(path)
    if cached is not None:
        print(
            f"[{done}/{total}] regq {label} seed={seed} RESUMED {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return cached

    X, y, gt = make_highdim_synthetic(p=p, seed=seed)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    rows: list[dict] = []
    for k in ks:
        selector, runtime_s = _fit_selector(REGQ_SPEC, "classification", k, seed, X_train, y_train)
        selector.redundancy = fast_pearson_penalty  # matches the baseline fast path
        order = list(selector.selected_idx_)
        metrics = recovery_metrics(order, gt)
        rank = ranking_scores(order, gt)
        recall_informative = recovery(order, gt).recall
        rows.append(
            {
                "kind": "synthetic",
                "dataset": label,
                "p": p,
                "operator": REGQ_SPEC.operator,
                "aggregation": REGQ_SPEC.aggregation,
                "k": k,
                "seed": seed,
                "recovery_f1": metrics["recovery_f1"],
                "noise_rate": metrics["noise_rate"],
                "recall_informative": recall_informative,
                "average_precision": rank.average_precision,
                "roc_auc": rank.roc_auc,
                "downstream_score": float("nan"),
                "runtime_s": runtime_s,
            }
        )

    df = pd.DataFrame(rows, columns=HIGHDIM_COLUMNS)
    _write_checkpoint(df, path)
    print(
        f"[{done}/{total}] regq {label} seed={seed} rows={len(df)} {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.highdim_regquotient", description=__doc__)
    parser.add_argument("--ps", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--out", type=str, default="results/highdim_regq.parquet")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = Path(args.out)
    checkpoint_dir = out.parent / "_highdim_regq_shards"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shards = [(p, seed) for p in args.ps for seed in args.seeds]
    total = len(shards)
    print(
        f"reg_quotient arm: {total} synthetic shards "
        f"({len(args.ps)} p-values x {len(args.seeds)} seeds)",
        flush=True,
    )
    frames = [
        _regq_shard(p, seed, args.ks, checkpoint_dir, i, total)
        for i, (p, seed) in enumerate(shards, start=1)
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(HIGHDIM_COLUMNS[:7], kind="stable").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} reg_quotient rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
