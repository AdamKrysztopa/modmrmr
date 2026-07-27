# Rust extension: decision record

Date: 2026-07-27
Branch: `perf/profiling-and-rust`

Machine: Apple Silicon (Darwin 25.5.0), Python 3.12, single interpreter for both
the baseline and the post-optimization sweep. Timings are machine-specific; only
the before/after *ratios* transfer.

## Gate criteria

Rust is justified only if all three hold for a single kernel:

1. **Amdahl** — accounts for >=50% of end-to-end time.
2. **Superlinear and irreducible** — fitted scaling exponent >=1.5 with no
   vectorized formulation left unexploited.
3. **Not third-party-bound** — `tree_r2` and `relieff` excluded by construction.

## Measured speedups (Phase 2)

Reproduce with `uv run python -m benchmarks.profiling.compare`.

### `driver_scaling` — `as_penalty_matrix` vs `p`, `pearson_abs` fixed

This grid isolates the O(p^2) dispatch overhead, so it shows the matrix fast
path (Task 9) in isolation.

```
     scorer  data_kind   n   p  median_s_before  median_s_after      speedup
pearson_abs continuous 500  25         0.074318        0.000069  1080.348795
pearson_abs continuous 500  50         0.287912        0.000079  3657.932244
pearson_abs continuous 500 100         1.169715        0.000130  8977.641858
pearson_abs continuous 500 200         4.816086        0.000341 14111.335050
pearson_abs continuous 500 400        19.529719        0.001131 17270.191018
```

### `end_to_end` — full `MRMRSelector.fit`

```
             scorer  data_kind    n   p  median_s_before  median_s_after    speedup
               gcmi continuous  500  50         0.533767        0.018678   28.577549
               gcmi continuous  500 200         8.626668        0.057389  150.317876
               gcmi   discrete 2000 200        12.344756        0.083100  148.552577
          mixed_ksg   discrete  500 200        60.237549       12.928016    4.659458
          mixed_ksg   discrete 2000  50        21.995303        2.935993    7.491607
          mixed_ksg continuous 2000 200        81.898690       40.884166    2.003188
mutual_info_sklearn continuous  500 200        45.331301       43.155562    1.050416
mutual_info_sklearn continuous 2000 200       135.160755      133.785925    1.010276
        pearson_abs continuous  500 200         4.669882        0.025660  181.988365
        pearson_abs continuous 2000 200         5.108806        0.028002  182.443772
```

### `scorer_scaling` — per-pair `score_pair` cost

Per-pair cost is unchanged for most scorers (the Phase 2 work was at the driver
level, not inside `score_pair`). The exceptions are the two scorers whose
per-pair cost contained the O(n^2) tie loop removed in Task 10, and only on
discrete data where ties actually occur:

```
scorer        data_kind   max speedup
mixed_ksg     discrete       6.99
ami_adaptive  discrete       6.32
```

Every other (scorer, data_kind) pair sits within noise of 1.0x, as expected.

## Fitted exponents after optimization

`log(t) = a + b*log(n)`, per (scorer, data_kind), from `scorer_scaling_after.csv`.
Worst (largest) exponent per scorer:

```
             scorer  worst exponent
             kde_mi        1.387772
      distance_corr        1.182111
       ami_adaptive        1.109475
          mixed_ksg        1.000333
        catt_knn_mi        0.919199
mutual_info_sklearn        0.869738
          copula_mi        0.704517
       spearman_abs        0.639262
         bspline_mi        0.642027
                rdc        0.570081
               gcmi        0.568262
        pearson_abs        0.129995
            relieff             NaN  (1 measured point)
            tree_r2             NaN  (1 measured point)
```

**Nothing reaches the 1.5 threshold.** The highest is `kde_mi` at 1.39, and it
holds a negligible share of end-to-end time. `tree_r2` and `relieff` have too
few measured points to fit — the remaining cells exceeded the per-cell budget —
but both are third-party-fit-bound and excluded by construction regardless.

## Verdict

```
             scorer  time_share  dominates  exponent  superlinear  own_code  passes_gate
mutual_info_sklearn    0.666908       True  0.869738        False      True        False
          mixed_ksg    0.331594      False  1.000333        False      True        False
               gcmi    0.001089      False  0.568262        False      True        False
        pearson_abs    0.000410      False  0.129995        False      True        False

GATE FAILS — no kernel is simultaneously dominant, superlinear, and not
third-party-bound. Rust is not justified.
```

**Scope of the time_share column.** The `end_to_end` grid runs four scorers
(`gcmi`, `mixed_ksg`, `mutual_info_sklearn`, `pearson_abs`), so `time_share` is
a share *within that grid*, not across all fourteen registered scorers. This
does not change the verdict — condition 2 fails independently for every scorer
measured, including the ones outside the grid: the slowest per-pair scorer is
`kde_mi` (0.206s/pair at n=10,000, 13x `mutual_info_sklearn`) and its exponent
is 1.39, still below the 1.5 threshold. `tree_r2` and `relieff` are far more
expensive still (0.072s and 0.009s per pair at n=200, where the others are
sub-millisecond) and exceeded the per-cell budget at every larger n, which is
why only 3 of 18 cells were measured for each — but both are third-party-bound
and excluded by construction.

**Decision:** Rust is not justified; close the question.

**Reasoning:** The gate fails on condition 2 for every scorer, and the failure is
structural rather than marginal. `mixed_ksg` was the kernel that motivated this
investigation — its O(n^2) coincidence-count loop was the one genuinely
superlinear piece of our own code. Task 10 replaced it with an `np.unique` pass
guarded by an eps-separation check, and its fitted exponent is now 1.00: the
quadratic term Rust was meant to attack no longer exists. What remains dominant
is `mutual_info_sklearn` at 66.7% of end-to-end time, and it fails the gate twice
over — it scales *sublinearly* (0.87), so a constant-factor rewrite is the wrong
lever, and its cost is inside scikit-learn's KSG estimator, which a Rust
extension in this package cannot touch. Note also that its share is high
*because* Phase 2 succeeded elsewhere: it is the one end-to-end configuration
that did not speed up (1.01x), while `pearson_abs` and `gcmi` improved by
150-182x and the pairwise driver by up to 17,000x at p=400. The profile is now
dominated by third-party code and by measures that are already effectively
linear.

The honest summary is that the expensive part was never the language. It was
`p(p-1)/2` Python-level dispatches around cheap statistics, and one accidental
O(n^2) loop. Both are gone, in pure NumPy, with every fast path gated by
`assert_parity` against the reference implementation.

A gate failure is a successful outcome: it means the cheap wins were the whole
story and a toolchain dependency would have bought little.

## What would reopen the question

This decision is evidence-bound, not permanent. Re-run
`uv run python -m benchmarks.profiling.run --benchmark all` and
`uv run python -m benchmarks.profiling.compare` if any of these change:

- A new scorer lands with a fitted exponent >=1.5 that takes a dominant share.
- `mixed_ksg` becomes dominant at a workload shape not covered by this grid
  (its exponent is 1.00 here, so it would have to be constant-factor cost, which
  is a vectorization question before it is a Rust question).
- The `mutual_info_sklearn` dependency is replaced by an in-repo estimator,
  which would move that 66.7% share from third-party code to our own.
