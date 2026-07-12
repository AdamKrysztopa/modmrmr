"""Deep-dive: is MI a bad relevance measure, or is sklearn's MI estimator the problem?

Compares MI-family relevance estimators head-to-head on the classification golden sets
(where the factorial's mutual_info_classif looked weak on average), holding redundancy=
pearson_abs, operator=difference, aggregation=mean fixed. Separates two confounds:
  - estimator: sklearn discrete (mutual_info_classif) vs sklearn-regression (mutual_info_sklearn,
    n=3) vs KSG-regression (ksg_mi / "cross_ami", n=8) vs Gaussian-copula (gcmi)
  - neighbor count: mutual_info_classif at n_neighbors=3 vs 8
Baseline: pearson_abs (linear). Metric: recovery F1/noise vs known ground truth.
Verbose + checkpoints analysis/mi_estimator_probe.csv.
"""

import time

import pandas as pd

from mechanism.datasets import load_mechanism_dataset
from mechanism.recovery import ranking_scores, recovery
from modmrmr.core.estimator import MRMRSelector
from modmrmr.core.scorers import get_scorer, register_scorer

# instances aren't accepted directly as relevance (names only), so register a
# neighbor-count variant of the discrete-target classif MI to isolate the n_neighbors effect.
register_scorer("mi_classif_n8", get_scorer("mutual_info_classif").with_neighbors(8))

CLF_SETS = [
    "nonlin_redundancy_clf",  # nonlinear
    "xor",  # nonlinear/interaction
    "linear_control_clf",  # linear
    "redundant_blocks_clf",  # linear, redundancy-rich
    "quotient_trap_clf",  # linear, W->0 trap
]


# relevance estimators to compare (name shown -> how to build)
def _relevances():
    yield "pearson_abs (baseline)", "pearson_abs"
    yield "MI sklearn-classif n3", "mutual_info_classif"
    yield "MI sklearn-classif n8", "mi_classif_n8"
    yield "MI sklearn-reg n3", "mutual_info_sklearn"
    yield "KSG-reg n8 (=AMI/cross_ami)", "ksg_mi"
    yield "GCMI (copula)", "gcmi"


rows = []
for ds in CLF_SETS:
    X, y, task, gt = load_mechanism_dataset(ds)
    dep = gt.dependence
    print(f"\n[{ds}] dependence={dep} shape={X.shape}", flush=True)
    for label, rel in _relevances():
        for k in [5, 10]:
            t = time.time()
            try:
                sel = MRMRSelector(
                    n_features=k,
                    relevance=rel,
                    redundancy="pearson_abs",
                    operator="difference",
                    aggregation="mean",
                    task=task,
                    random_state=0,
                )
                sel.fit(X, y)
                idx = list(sel.selected_idx_)
                r = recovery(idx, gt)
                rk = ranking_scores(idx, gt)
                rows.append(
                    dict(
                        dataset=ds,
                        dependence=dep,
                        estimator=label,
                        k=k,
                        f1=r.f1,
                        precision=r.precision,
                        recall=r.recall,
                        noise_rate=r.noise_rate,
                        average_precision=rk.average_precision,
                    )
                )
                print(
                    f"    {label:28s} k={k:>2} F1={r.f1:.3f} noise={r.noise_rate:.3f} "
                    f"AP={rk.average_precision:.3f} ({time.time() - t:.1f}s)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    {label:28s} k={k:>2} FAILED {type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
    pd.DataFrame(rows).to_csv("analysis/mi_estimator_probe.csv", index=False)

pd.DataFrame(rows).to_csv("analysis/mi_estimator_probe.csv", index=False)
print(f"\nWROTE analysis/mi_estimator_probe.csv ({len(rows)} rows)", flush=True)
