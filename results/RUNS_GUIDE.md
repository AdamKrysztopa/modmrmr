# mRMR Studies — Runs Guide

Single source of truth for every study run in the mRMR design-space investigation:
what was run, what it produced, where it lives, and how to reproduce it. This file
is the **index + reproduction guide**.

Determinism everywhere: seeds only (`np.random.default_rng(seed)` / threaded `random_state`).
Toolchain: `uv` (never bare `python`/`pytest`/`ruff`).

## Study index

| # | Study | Driver | Grid / scope | Recovery? | Status |
|---|---|---|---|---|---|
| 1 | Full factorial design-space | `mechanism/run_factorial_fast.py` | 180 specs × 19 datasets × ks×thresholds×seeds × 4 stop modes = 117,180 rows | golden only | ✅ committed `c750247` |
| 2 | High-dim operator | `analysis/highdim_operator_study.py` | pearson×pearson × 4 operators; madelon-style recovery + real downstream | madelon_synth only | ✅ committed `b611bb9` |
| 3 | MI-estimator probe | `analysis/mi_estimator_probe.py` | 6 MI/relevance variants × 5 clf golden sets × k | yes (golden) | ✅ committed `b27b0cf` |
| 4 | MI-methods comparison | `mechanism/run_mi_comparison.py` | 12 MI estimators × 15 golden × ks × seeds = 2844 rows (relevance-only) | yes (golden) | ✅ committed `05328fc` |

## Engines (how selection is actually run)

- **Reference:** `mechanism/factorial_protocol.py::run_factorial_grid` — the audited grid engine
  (leakage-free train/test/val splits; per-cell error rows; 4 stop modes). The oracle.
- **Fast:** `mechanism/fast_factorial_protocol.py::run_fast_factorial_grid` /
  `run_fast_downstream_only_grid` — memoizes the relevance vector + redundancy matrix per
  (split, relevance, redundancy) and injects them into a real `MRMRSelector`. **Proven
  byte-identical** to the reference (`tests/mechanism/test_fast_equivalence.py`; numerical
  review confirmed uncovered combos too). Speedup ~1.5× (downstream RF fits dominate).
- **CLI:** `mechanism/run_factorial_fast.py` — verbose per-shard progress (`print(flush=True)`
  + `joblib.Parallel(verbose=10)`), per-shard parquet checkpoints under `results/_shards/`,
  resumable (`--fresh` to ignore checkpoints). `--list` standalone.
- **Stop modes:** `fixed_k`, `threshold`, `val_fixed_k`, `val_threshold` (validation-selected
  operating point chosen on a held-out val split, evaluated on test).

Operational lessons: loky workers cold-spawn 1–2 min — a background run
showing 0 workers early is **not** dead; never stack two loky runs (contention kills them);
heavy scripts must be verbose + checkpointed.

---

## Study 1 — Full factorial design-space

**Purpose:** map recovery + downstream across the whole mRMR design space.

**Axes (180 specs):** relevance {pearson, spearman, f, mi, dcor} × redundancy {pearson_abs,
spearman_abs, mutual_info_sklearn, distance_corr} × operator {difference, multiplicative,
quotient} × aggregation {mean, max, sum}. `mi` resolves to sklearn `mutual_info_classif` /
`mutual_info_regression` (**n_neighbors=3** — see Study 3 caveat).

**Datasets (19):** 15 synthetic golden (ground truth → recovery) + 4 benchmark
(downstream-only): `breast_cancer` (real), `diabetes` (real), `friedman1` (synthetic),
`synthetic_clf` (synthetic). ks {1,2,3,5,8,10}; thresholds {0,0.05,0.1,0.2}; seeds {0,1,2}.

**Outputs:**
| file | contents | git |
|---|---|---|
| `results/factorial.parquet` | all 117,180 rows, every metric per cell | **local** (gitignored, 1.5 MB) |
| `results/factorial_summary.csv` | per (spec, dependence) means (golden only) | committed |
| `results/factorial_decision_guide.csv` | best spec by F1 / by noise per dataset | committed |
| `results/factorial_benchmark_summary.csv` | real-benchmark downstream by spec | committed |
| `results/_shards/*.parquet` (57) | per (dataset,seed) checkpoints | local |
| `paper/artifacts/*.png` (35) | figures | local (gitignored) |

**Reproduce:**
```
uv run python -m mechanism.run_factorial_fast --specs all \
  --ks 1 2 3 5 8 10 --thresholds 0.0 0.05 0.1 0.2 --seeds 0 1 2 \
  --out results/factorial.parquet --figures-dir paper/artifacts --jobs -1
```
Wall-clock ≈ 29 min on 12 cores. 0 errors.

**Headline:** the **relevance measure dominates** (match to structure: MI/dcor nonlinear,
Pearson linear; best score×penalty = MI-reg/dcor × Pearson/dcor). Operator is secondary; the
**quotient is the worst operator for recovery** (W→0 noise-crowning, never a top spec) yet
**ties multiplicative on real downstream** (0.676). Validation-selected fixed-k gives the best
recovery F1 (0.519) at compact k≈4.4.

**Caveats:** recovery only exists for the 15 golden sets; the "real 24-set benchmark" is only
2 real + 2 synthetic here (rest need network/files — see Study 2 for network follow-up).

---

## Study 2 — High-dimensional operator study

**Purpose:** does the operator ordering hold at high dimension and on real data?

**Config:** fixed measure `pearson_abs × pearson_abs` (dcor/MI are O(n²)/O(p²) — infeasible at
p in the thousands), across operators {difference, multiplicative, quotient} + multiplicative+max.

**Parts:** (A) recovery on a **madelon-style `make_classification` set with known ground truth**
(p=500: 5 informative + 15 redundant + 480 noise — the *real* OpenML madelon does not
distribute true-feature identities, hence a stand-in). (B) downstream on real OpenML sets
(madelon, isolet completed).

**Outputs:** `results/highdim_operator_study.csv` (committed); script
`analysis/highdim_operator_study.py`.

**Reproduce:** `uv run python analysis/highdim_operator_study.py` (run from repo root; heavy).

**Headline:** the quotient's W→0 collapse **amplifies with dimension** — on the p=500 set it
recovers ~0 (F1≈0, noise 0.65) while multiplicative gets F1 0.53 at noise 0.20; multiplicative
also dominates **real** madelon downstream (0.86 vs ~0.62). **Refinement:** multiplicative +
**mean** > multiplicative + **max** at high dim → the contribution is *the multiplicative
operator's bounded veto*, not "multiplicative+max" specifically.

**Caveats / not done:** riboflavin (p=4088), gisette (5000), arcene (10000) are **impractical**
with the current mRMR (~39 min/fit at p=4088) — excluded. isolet shows operators are close
(difference edges ahead) → the multiplicative edge is scoped to redundancy-rich data.

---

## Study 3 — MI-estimator probe (is "MI bad" the measure or the estimator?)

**Purpose:** the factorial's "MI-classif is worst" used **one** MI estimator (sklearn kNN,
n=3). Separate two confounds — estimator implementation and neighbor count.

**Config:** relevance ∈ {pearson_abs (baseline), MI sklearn-classif n=3, MI sklearn-classif
n=8, MI sklearn-reg n=3, KSG-reg n=8 (=`ksg_mi`=`cross_ami_score`), GCMI} × 5 classification
golden sets × k {5,10}; redundancy=pearson_abs, operator=difference, aggregation=mean, seed 0.

**Outputs:** `analysis/mi_estimator_probe.csv`; script `analysis/mi_estimator_probe.py`.

**Reproduce:** `uv run python -m analysis.mi_estimator_probe`

**Headline:** **"MI is worst" is largely an artifact of sklearn's default n_neighbors=3.** On
`linear_control_clf`, bumping the *same* estimator 3→8 lifts ranking AP 0.625→**1.000**
(matches Pearson). On nonlinear sets the estimator choice is irrelevant (all tie). On
`redundant_blocks_clf`, GCMI recovers to Pearson's AP; on `quotient_trap_clf` a small residual
MI-measure cost remains even for GCMI. So MI, with a tuned-neighbor kNN or a Gaussian-copula
estimator, is competitive with Pearson on linear classification.

**Caveats:** `cross_ami_score` = `ksg_mi` = the same n=8 KSG engine — **no distinct AMI
estimator exists** in the repo. All KSG/GCMI scorers are
regression/continuous-target engines. *(Filled by Study 4: `ami_adaptive` is now a genuine
data-adaptive-k MI estimator, superseding the `cross_ami_score` alias.)*

---

## Corrections feeding Phase E (do not write the paper without these)

1. **The "MI-classif worst" claim is RESOLVED by Study 4, not just qualified by Study 3.**
   Study 3 suggested it was a sklearn-default-neighbor artifact; Study 4's 12-estimator sweep
   under the leakage-free split shows that read was itself a **full-data-fit artifact** — the
   n3-vs-n8 effect is dataset-dependent and *reverses* on `linear_control_clf`. The
   MI-**estimator** axis turns out to be second-order regardless (0.066 AP spread across all
   12 estimators), and the maligned `mi_reg_k3` is in fact **best-in-class on nonlinear
   recovery** (AP 0.660) and mid-pack overall. No estimator gets within 0.05 of Pearson on
   linear structure, so the P3 gate failed and the incumbent stands (see Study 4).
2. **Frame the operator contribution as "the multiplicative operator"** (bounded redundancy
   veto), NOT "multiplicative+max" — max did not generalize (Study 2).
3. **Recovery ≠ downstream** — the quotient is worst for recovery but ties for downstream;
   state both.

## Study 4 — MI-methods comparison

**Purpose:** Study 1's factorial used a single MI estimator (sklearn kNN, n=3) for the `mi`
relevance measure. Study 3 raised the possibility that its apparent weakness was an
estimator-default artifact. Study 4 settles it with a broad sweep: vary **only** the MI
estimator in the relevance slot and check whether a better one changes the main study's
conclusions.

**Config/Axes:** 12 MI estimators (6 newly implemented — `mixed_ksg`, `ami_adaptive`,
`bspline_mi`, `kde_mi`, `copula_mi`, plus kNN-neighbor variants at n∈{3,5,8,16}; plus the
incumbents `mi_reg_k3`, `ksg_mi`, `gcmi`) × 15 golden datasets × ks {1,2,3,5,8,10} × seeds
{0,1,2}; redundancy=`pearson_abs`, operator=`difference`, aggregation=`mean` held fixed;
leakage-free 70/30 split (the project's standard protocol). Driver:
`mechanism/run_mi_comparison.py` (`run_mi_comparison_grid`).

**Outputs:**
| file | contents | git |
|---|---|---|
| `results/mi_comparison.parquet` | all 2,844 rows, every metric per cell | local (gitignored) |
| `results/mi_comparison_summary.csv` | per-estimator × dependence-class means | committed |
| `results/mi_comparison_winner.csv` | winner selection table | committed |
| `paper/artifacts/mi_ranking_leaderboard.png` | estimator leaderboard figure | local (gitignored) |
| `paper/artifacts/mi_ap_by_dependence.png` | AP by dependence class figure | local (gitignored) |

**Reproduce:**
```
uv run python -m mechanism.run_mi_comparison --estimators all \
  --ks 1 2 3 5 8 10 --seeds 0 1 2 \
  --out results/mi_comparison.parquet --figures-dir paper/artifacts --jobs -1
```
2,844 rows, 0 errors.

**Headline:** the **MI-estimator axis is a wash** — the field spans only 0.066 mean ranking-AP
(0.559–0.625), no estimator dominates. `copula_mi` is the winner by the no-nonlinear-regression
rule (0.620, beats incumbent `mi_reg_k3` 0.612 by only 0.008, worse on linear). Study 3's
"sklearn-n3 is the confound" does **not** survive the leakage-free split — it was a full-data-fit
artifact that reverses on `linear_control_clf`; the incumbent `mi_reg_k3` has the best nonlinear
AP of all 12 estimators (0.660). The P3 re-run gate **failed**: no MI estimator comes within 0.05
of `pearson_abs` on linear golden sets (0.731 vs best MI 0.600) — so the incumbent stands, and
the main 180-spec study is **not** re-run.

**Caveats:** `ami_adaptive` supersedes the old `cross_ami_score` (= KSG alias) noted as a gap in
Study 3 — there is now a genuine data-adaptive-k MI estimator, not just a KSG re-label. Recovery
over the golden sets is the discriminating metric; downstream (real-model accuracy) is flatter
still (0.776–0.821 across all estimators) and secondary.

---

## All-runs reproduce cheat-sheet

```
# Study 1 (full factorial, ~29 min / 12 cores)
uv run python -m mechanism.run_factorial_fast --specs all \
  --ks 1 2 3 5 8 10 --thresholds 0.0 0.05 0.1 0.2 --seeds 0 1 2 \
  --out results/factorial.parquet --figures-dir paper/artifacts --jobs -1

# Study 2 (high-dim operator; heavy — p>=4000 sets impractical)
uv run python analysis/highdim_operator_study.py

# Study 3 (MI-estimator probe; fast)
uv run python -m analysis.mi_estimator_probe

# Study 4 (MI-comparison; ~1 min / 12 cores)
uv run python -m mechanism.run_mi_comparison --estimators all \
  --ks 1 2 3 5 8 10 --seeds 0 1 2 \
  --out results/mi_comparison.parquet --figures-dir paper/artifacts --jobs -1

# Study 5 (hardening: 10-seed factorial + high-dim p-sweep + baselines + significance/cost)
uv run python -m mechanism.run_factorial_fast --specs all \
  --ks 1 2 3 5 8 10 --thresholds 0.0 0.05 0.1 0.2 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --out results/factorial.parquet --figures-dir paper/artifacts --jobs -1   # ~55 min / 12 cores
uv run python -m analysis.highdim_study --ps 500 1000 2000 5000 10000 \
  --seeds 0 1 2 --ks 5 10 20 --out results/highdim_study.parquet            # fast pearson path
uv run python -m mechanism.run_baseline_comparison                          # external baselines
uv run python -m analysis.hardening_outputs                                 # stats + cost + summary CSVs
```

## Study 5 — Hardening (paper-ready)

- **Purpose:** reviewer-proof the paper — 10 seeds, Friedman/Nemenyi significance,
  high-dim p-sweep to p=10,000 (vectorized Pearson), external baselines, cost.
- **Outputs:** `results/factorial.parquet` (390,600 rows, 10 seeds);
  `results/highdim_study.parquet` + `_summary.csv`; `results/baseline_comparison.csv` +
  `_summary.csv`; `results/stats_operators*.csv`,
  `results/stats_measure_families*.csv` (incl. `_by_dependence.csv`),
  `results/cost_summary.csv`, `results/cost_vs_p.csv`; CD diagrams
  `paper/artifacts/cd_operators.pdf`, `cd_measures.pdf`.
- **Headline (3 unsaturated metrics over the 5 informative, k=20):** quotient surfaces
  only ~1.3 of 5 informative (recall 0.27) with noise rising 0.667→0.783 as p 500→10k;
  multiplicative(mean) finds 4/5 (recall 0.80) at 2.4× less noise (0.20) than difference
  (same 0.80 recall, 0.47 noise) and best/stable AP (~0.40); multiplicative+MAX degrades
  with p (recall 0.73→0.53) — mean aggregation, not max, is load-bearing. Fixed-k F1 is
  recall-ceiling'd (appendix only). Golden significance: operator noise-rate p<0.0001,
  operator F1 p=0.549 (expected); measure effect significant on linear (p=0.002),
  underpowered on nonlinear (4 datasets, p=0.472).
- **Caveats:** nonlinear golden-set count (4) underpowers the nonlinear half; gisette
  skipped (sparse-ARFF loader limitation).
