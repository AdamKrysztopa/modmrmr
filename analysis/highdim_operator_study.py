"""High-dim operator study: pearson_abs x pearson_abs across {difference, multiplicative,
quotient, mult+max}. Part A = recovery on a madelon-style synthetic set (known ground truth).
Part B = downstream on real OpenML high-dim sets. Verbose; writes
results/highdim_operator_study.csv.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mechanism.factorial import SelectorSpec, build_selector
from mechanism.ground_truth import GroundTruth
from mechanism.protocol import _downstream
from mechanism.recovery import ranking_scores, recovery

SPECS = {
    "difference": SelectorSpec("pearson", "pearson_abs", "difference", "mean"),
    "multiplicative": SelectorSpec("pearson", "pearson_abs", "multiplicative", "mean"),
    "quotient": SelectorSpec("pearson", "pearson_abs", "quotient", "mean"),
    "mult+MAX": SelectorSpec("pearson", "pearson_abs", "multiplicative", "max"),
}
KS = [5, 10, 20]


def log(msg):
    print(msg, flush=True)


rows = []

# ---- Part A: madelon-style synthetic recovery (known ground truth) ----
log("=== PART A: recovery (madelon-style, p=500, 5 informative + 15 redundant + 480 noise) ===")
Xa, ya = make_classification(
    n_samples=2600,
    n_features=500,
    n_informative=5,
    n_redundant=15,
    n_repeated=0,
    n_clusters_per_class=2,
    class_sep=1.0,
    shuffle=False,
    random_state=0,
)
Xa = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(500)])
ya = pd.Series(ya, name="target")
gt = GroundTruth(
    informative=tuple(range(5)),
    codependent=(tuple(range(5, 20)),),
    noise=tuple(range(20, 500)),
    dependence="mixed",
    n_features=500,
)
for name, spec in SPECS.items():
    for k in KS:
        t = time.time()
        sel = build_selector(spec, "classification", k, None, 0)
        sel.fit(Xa, ya)
        idx = list(sel.selected_idx_)
        r = recovery(idx, gt)
        rk = ranking_scores(idx, gt)
        rows.append(
            dict(
                part="recovery",
                dataset="madelon_synth",
                operator=name,
                k=k,
                precision=r.precision,
                recall=r.recall,
                f1=r.f1,
                noise_rate=r.noise_rate,
                average_precision=rk.average_precision,
                roc_auc=rk.roc_auc,
                downstream=np.nan,
            )
        )
        log(
            f"  [A] {name:15s} k={k:>2} F1={r.f1:.3f} noise={r.noise_rate:.3f} "
            f"AP={rk.average_precision:.3f} ({time.time() - t:.1f}s)"
        )

# ---- Part B: real OpenML high-dim downstream ----
from benchmarks.datasets import load_dataset  # noqa: E402

REAL = ["madelon", "isolet", "riboflavin", "gisette", "arcene"]
log("\n=== PART B: real OpenML high-dim downstream (pearson x pearson) ===")
for ds in REAL:
    try:
        t0 = time.time()
        X, y, task = load_dataset(ds)
        log(f"  [B] loading {ds}: {X.shape} task={task} ({time.time() - t0:.1f}s)")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        for name, spec in SPECS.items():
            for k in KS:
                t = time.time()
                sel = build_selector(spec, task, k, None, 0)
                sel.fit(Xtr, ytr)
                idx = list(sel.selected_idx_)
                ds_score = _downstream(Xtr, Xte, ytr, yte, idx, task)
                rows.append(
                    dict(
                        part="downstream",
                        dataset=ds,
                        operator=name,
                        k=k,
                        precision=np.nan,
                        recall=np.nan,
                        f1=np.nan,
                        noise_rate=np.nan,
                        average_precision=np.nan,
                        roc_auc=np.nan,
                        downstream=ds_score,
                    )
                )
                log(f"      {name:15s} k={k:>2} downstream={ds_score:.4f} ({time.time() - t:.1f}s)")
        # checkpoint after each real dataset
        pd.DataFrame(rows).to_csv("results/highdim_operator_study.csv", index=False)
        log(f"  [B] {ds} done, checkpointed ({time.time() - t0:.1f}s total)")
    except Exception as exc:  # noqa: BLE001
        log(f"  [B] SKIP {ds}: {type(exc).__name__}: {str(exc)[:160]}")

pd.DataFrame(rows).to_csv("results/highdim_operator_study.csv", index=False)
log(f"\nWROTE results/highdim_operator_study.csv ({len(rows)} rows)")
sys.exit(0)
