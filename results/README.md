# results/ — artifact notes

## High-dimensional study

`highdim_study.parquet` is the **canonical** high-dim artifact referenced by the
paper's Figure 3 / Study 5. It is a three-arm concatenation:

1. **Synthetic p-sweep** (10 seeds, `p in {500, 1000, 2000, 5000, 10000}`) and
2. **Real high-dim sets** (3 seeds: madelon, isolet, riboflavin, arcene — gisette
   is skipped: sparse-ARFF loader limitation)

   both produced by `analysis.highdim_study`, plus

3. The **regularized-quotient arm** `score = rel / (agg_red + eps)`
   (`kind == "reg_quotient"`), produced by `analysis.highdim_regquotient` and
   merged in to answer whether additive regularization rescues the quotient
   operator at high dimension.

`_highdim_regq_shards/` holds the intermediate per-`(dataset, seed)` checkpoint
files for that third arm — resumable progress state, not a release artifact.

### Synthetic downstream (predictive) score

`highdim_study.parquet`'s `downstream_score` column is **NaN for every
`kind == "synthetic"` row by construction** — the original driver that built the
synthetic p-sweep arm never trained a held-out model on the operator's selection
(only recovery-against-ground-truth metrics are scored there). The synthetic
downstream numbers quoted in the paper (Sec.~5 / Fig. `fig_highdim_downstream`)
come from a separate, purpose-built artifact instead of a parquet merge:

- **`highdim_downstream_synth.csv`** is the **canonical carrier** of the
  synthetic-arm downstream score. It reproduces the exact selection each
  operator (`difference`, `multiplicative`, `quotient`, `reg_quotient`) makes on
  the train split of each `(p, seed)` synthetic set, then scores a held-out
  random forest (`mechanism.protocol._downstream`) on the test split — same
  models/metrics discipline as the real-arm downstream score in
  `highdim_study.parquet`. Produced by `analysis.highdim_downstream_synth`
  (10 seeds x 5 `p` values x 4 operators = 200 rows); the operator column keeps
  the raw token `multiplicative` (see "Display names" below).
- `_downstream_synth_shards/` holds the intermediate per-`(p, seed)` checkpoint
  files — resumable progress state, not a release artifact.

Do not hand-merge this CSV's numbers into `highdim_study.parquet`; treat the two
as separate artifacts with a shared operator vocabulary, cross-referenced by the
reproduction commands in `paper/sections/99-appendix.tex`.

## Display names vs. raw tokens

Every results CSV keeps the **raw** spec/operator token `multiplicative` in its
`operator`/`spec` columns (e.g. `factorial_leaderboard.csv`,
`highdim_downstream_synth.csv`, `stats_operators_nemenyi_*.csv`) — CSVs are never
rewritten to carry a display name. The paper calls this operator "the gate" in
prose and in every *generated* table/figure; the token -> display-name
mapping (`multiplicative` -> `gate`, `reg_quotient` -> `reg. quotient`) lives
in `analysis/display_names.py` and is applied only at render time by
`analysis.paper_tables` / `analysis.paper_figures` / `analysis.hardening_outputs`
(the `cd_operators.pdf` critical-difference diagram). See
`paper/artifacts/tab_appendix_leaderboard.tex`'s caption for the one-sentence
reader-facing explanation of the mapping.

## Operator significance (Nemenyi post-hoc)

The operator Friedman/Nemenyi test is reported for **two metrics**, and the
filenames say which:

- **`stats_operators_nemenyi_f1.csv`** — recovery F1, dataset-blocked. All
  pairwise p-values are non-significant (p > 0.5); F1 alone does not separate
  the operators. Identical to the historical `stats_operators_nemenyi.csv`
  (kept, unchanged, for backward compatibility — same content as the `_f1` copy).
- **`stats_operators_nemenyi_noise.csv`** — noise rate, mechanism-blocked (twin
  clf/reg pairs collapsed to one block). This is the load-bearing operator test
  cited in the paper: quotient-vs-gate p ~= 0.008, quotient-vs-difference
  p ~= 0.0002 (Sec. 5). Also the source of `cd_operators.pdf`.

Both are produced by `analysis.hardening_outputs`.

## `_archive/`

Superseded backups kept for traceability, not used by any script or the paper:

- `highdim_study_10seed.parquet` — an earlier 10-seed cut later superseded by
  the merged three-arm `highdim_study.parquet`.
- `highdim_study_3seed_backup.parquet` — a pre-merge 3-seed backup of the real-set
  arm.

Do not point new analysis at these; use `highdim_study.parquet`.
