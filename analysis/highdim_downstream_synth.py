"""Downstream (held-out predictive) score for the synthetic high-dim p-sweep.

``results/highdim_study.parquet`` grades the four operator variants (difference,
multiplicative, quotient, mult_max/reg_quotient) on the synthetic p-sweep purely by
recovery-against-planted-ground-truth metrics (F1, noise_rate, AP, ROC-AUC) —
``downstream_score`` is NaN for every synthetic row because the original driver never
trained a model on the operator's selection. This closes that gap: for each
(operator, p, seed) it reproduces the EXACT selection the operator makes on the train
split (same ``SelectorSpec``, same ``fast_pearson_penalty`` injection, same seed as
``analysis.highdim_study``), then trains the same RF the real-data arm uses
(``mechanism.protocol._downstream``: ``RandomForestClassifier(n_estimators=100,
random_state=0)``, scored by balanced accuracy) on the held-out test split.

Verbose and resumable: one flushed progress line per (p, seed) shard
(all 4 operators inside), each shard checkpointed to
``results/_downstream_synth_shards/`` so an interruption is recoverable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from analysis.highdim_regquotient import REGQ_SPEC
from analysis.highdim_study import (
    SPECS,
    _fit_selector,
    _load_checkpoint,
    _write_checkpoint,
    make_highdim_synthetic,
    recovery_metrics,
)
from mechanism.protocol import _downstream
from mechanism.recovery import ranking_scores, recovery

# The four operators the reviewer wants compared: the three original SPECS
# (mult_max is a mult/aggregation variant we don't need here) plus reg_quotient.
OPERATOR_SPECS = {
    "difference": SPECS["difference"],
    "multiplicative": SPECS["multiplicative"],
    "quotient": SPECS["quotient"],
    "reg_quotient": REGQ_SPEC,
}

DOWNSTREAM_COLUMNS = [
    "operator",
    "p",
    "seed",
    "k",
    "downstream_score",
    "recovery_f1",
    "noise_rate",
    "recall_informative",
    "average_precision",
    "roc_auc",
    "runtime_s",
]


def _shard(p: int, seed: int, k: int, checkpoint_dir: Path, done: int, total: int) -> pd.DataFrame:
    label = f"synthetic_p{p}"
    path = checkpoint_dir / f"{label}__{seed}.parquet"
    t0 = time.perf_counter()
    cached = _load_checkpoint(path)
    if cached is not None:
        print(
            f"[{done}/{total}] {label} seed={seed} RESUMED from checkpoint "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return cached

    X, y, gt = make_highdim_synthetic(p=p, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    rows: list[dict] = []
    for name, spec in OPERATOR_SPECS.items():
        selector, runtime_s = _fit_selector(spec, "classification", k, seed, X_train, y_train)
        idx = list(selector.selected_idx_)
        downstream = _downstream(X_train, X_test, y_train, y_test, idx, "classification")
        metrics = recovery_metrics(idx, gt)
        rank = ranking_scores(idx, gt)
        recall_informative = recovery(idx, gt).recall
        rows.append(
            {
                "operator": name,
                "p": p,
                "seed": seed,
                "k": k,
                "downstream_score": downstream,
                "recovery_f1": metrics["recovery_f1"],
                "noise_rate": metrics["noise_rate"],
                "recall_informative": recall_informative,
                "average_precision": rank.average_precision,
                "roc_auc": rank.roc_auc,
                "runtime_s": runtime_s,
            }
        )

    df = pd.DataFrame(rows, columns=DOWNSTREAM_COLUMNS)
    _write_checkpoint(df, path)
    print(
        f"[{done}/{total}] {label} seed={seed} rows={len(df)} {time.perf_counter() - t0:.1f}s "
        f"(elapsed shard time; running ETA printed by caller)",
        flush=True,
    )
    return df


def run_downstream_sweep(
    ps: list[int], seeds: list[int], k: int, *, checkpoint_dir: Path | str
) -> pd.DataFrame:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shards = [(p, seed) for p in ps for seed in seeds]
    total = len(shards)
    print(
        f"running {total} downstream-synth shards ({len(ps)} p-values x {len(seeds)} seeds, k={k})",
        flush=True,
    )
    t_start = time.perf_counter()
    frames = []
    for i, (p, seed) in enumerate(shards, start=1):
        frames.append(_shard(p, seed, k, checkpoint_dir, i, total))
        elapsed = time.perf_counter() - t_start
        rate = elapsed / i
        eta = rate * (total - i)
        print(
            f"  progress: {i}/{total} shards done, elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
            flush=True,
        )
    if not frames:
        return pd.DataFrame(columns=DOWNSTREAM_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.highdim_downstream_synth", description=__doc__)
    parser.add_argument("--ps", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=str, default="results/highdim_downstream_synth.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = Path(args.out)
    checkpoint_dir = out.parent / "_downstream_synth_shards"

    df = run_downstream_sweep(args.ps, args.seeds, args.k, checkpoint_dir=checkpoint_dir)
    if df.empty:
        print("downstream-synth sweep produced no rows")
        return 1

    df = df.sort_values(["p", "operator", "seed"], kind="stable").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}", flush=True)

    pivot = df.pivot_table(index="operator", columns="p", values="downstream_score", aggfunc="mean")
    print("\nmean downstream_score by operator x p:", flush=True)
    print(pivot.to_string(), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
