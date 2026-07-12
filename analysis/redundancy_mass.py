"""Redundancy-mass diagnostic: does aggregate redundancy W concentrate near zero?

The separation proposition (paper Sec. 3) assumes that for a candidate feature
independent of the selected set, the aggregate redundancy W concentrates near
zero and anti-concentrates there -- which is what makes the quotient score
rel/W heavy-tailed, and hence what makes an argmax over p candidates pick noise.

This script measures W's empirical distribution on synthetic controls and on the
four real high-dimensional benchmarks, and reports the fraction of candidates in
the danger zone.

KNOWN CONFOUND (found empirically on the four real datasets, not a hypothetical):
a *fixed* absolute threshold on W is dominated by sample size n, not by how
correlated the features actually are. Under the null (a candidate independent of
the selected set), the sampling noise floor of |pearson(x, y)| shrinks as
~1/sqrt(n), so the aggregate redundancy W -- a mean of k such terms -- shrinks
with n too. A fixed threshold like W <= 0.05 therefore captures almost all of a
large-n dataset's mass (madelon, n=2600: 99.3%) and almost none of a small-n
dataset's mass (riboflavin, n=71: 0.01%) *regardless of the real correlation
structure in the data*. Concretely, riboflavin has the lowest fixed-threshold
mass of all four real datasets despite a large gate-vs-quotient win margin --
the opposite of what the fixed-threshold statistic would predict. That is not
evidence the separation mechanism is wrong; it is evidence the fixed-threshold
statistic mostly measures n. `mass_below_abs` is kept (fixed W <= 0.05) so this
confound stays visible rather than silently fixed; `mass_below_scaled` uses a
threshold that scales with the null noise floor (c / sqrt(n)) and is the
statistic that is actually comparable across datasets of different n. Do not
report `mass_below_abs` as a cross-dataset comparison without this caveat.

Emits incremental progress: a long silent run is indistinguishable from a hang.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.datasets import load_dataset
from modmrmr.core.scorers.base import fast_pearson_penalty

REAL_DATASETS = ["madelon", "isolet", "riboflavin", "arcene"]
DANGER_THRESHOLD_ABS = 0.05
NOISE_FLOOR_SCALE = 1.0  # c in threshold_scaled = c / sqrt(n)

# Synthetic arm (paper Sec. 3, assumption A1): mutually independent standard-normal
# columns, the controlled construction the theory section's numbers actually cite.
# Includes the paper's own (n, p) combos {200, 500} x {2000, 10000} as a subset.
SYNTHETIC_N_GRID = [100, 200, 500, 1000, 2000]
SYNTHETIC_P_GRID = [500, 1000, 2000, 5000, 10000]
SYNTHETIC_SEEDS = list(range(10))
SYNTHETIC_K_SELECTED = 10  # t = 10, matching Sec. 3's stated selected-set size


@dataclass(frozen=True)
class RedundancyMass:
    """Empirical summary of the aggregate-redundancy distribution over candidates.

    `mass_below_abs` uses a fixed W <= 0.05 threshold and is n-confounded (see
    module docstring); `mass_below_scaled` uses a threshold that scales with the
    null noise floor (NOISE_FLOOR_SCALE / sqrt(n)) and is comparable across n.
    """

    mass_below_abs: float
    mass_below_scaled: float
    w_median: float
    w_p05: float
    tail_slope: float


def log(msg: str) -> None:
    print(msg, flush=True)


def _aggregate_redundancy(X: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Mean |pearson| of every candidate against the selected set -- the mRMR W term."""
    pen = fast_pearson_penalty(pd.DataFrame(X)).to_numpy()
    candidates = np.setdiff1d(np.arange(X.shape[1]), selected)
    return pen[np.ix_(candidates, selected)].mean(axis=1)


def _tail_slope(scores: np.ndarray) -> float:
    """Hill exponent of the upper tail: smaller => heavier tail.

    Standard Hill estimator: sort descending, take the top k, and use the
    (k+1)-th largest order statistic -- the next value BELOW the top-k slice,
    not the smallest value inside it -- as the threshold. Averaging
    log(x_i / x_(k+1)) over the k values above threshold gives 1/alpha_hat.
    """
    positive = np.sort(scores[scores > 0])
    k = max(10, len(scores) // 10)
    if len(positive) < max(5, k + 1):
        return float("nan")
    top_k = positive[-k:]
    threshold = positive[-(k + 1)]
    return float(1.0 / np.mean(np.log(top_k / threshold)))


def _relevance(X: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """|corr(X_j, y)| for each candidate -- relevance is against the TARGET, not a feature."""
    xc = X[:, candidates] - X[:, candidates].mean(axis=0)
    yc = y - y.mean()
    denom = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
    return np.abs(xc.T @ yc) / np.maximum(denom, 1e-12)


def redundancy_mass(
    X: np.ndarray, y: np.ndarray, *, k_selected: int, random_state: int
) -> RedundancyMass:
    """Distribution of aggregate redundancy W over candidates, given a random selected set."""
    rng = np.random.default_rng(random_state)
    selected = rng.choice(X.shape[1], size=k_selected, replace=False)
    candidates = np.setdiff1d(np.arange(X.shape[1]), selected)
    w = _aggregate_redundancy(X, selected)
    rel = _relevance(X, np.asarray(y, dtype=float), candidates)
    quotient = rel / np.maximum(w, 1e-12)
    threshold_scaled = NOISE_FLOOR_SCALE / np.sqrt(X.shape[0])
    return RedundancyMass(
        mass_below_abs=float((w <= DANGER_THRESHOLD_ABS).mean()),
        mass_below_scaled=float((w <= threshold_scaled).mean()),
        w_median=float(np.median(w)),
        w_p05=float(np.quantile(w, 0.05)),
        tail_slope=_tail_slope(quotient),
    )


def _synthetic_independent_columns(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mutually independent standard-normal columns with one planted signal.

    Column 0 is the sole signal (y depends on it); every other column is
    independent of both y and every other column by construction, so the
    mechanism A1 claims (redundancy-against-a-random-selected-set concentrating
    near zero) is isolable -- there is no confounding correlation structure to
    explain it away, unlike the four real datasets (see module docstring).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 2.0 * X[:, 0] + rng.standard_normal(n)
    return X, y


def run_synthetic(
    out_csv: Path,
    *,
    k_selected: int = SYNTHETIC_K_SELECTED,
    n_grid: list[int] = SYNTHETIC_N_GRID,
    p_grid: list[int] = SYNTHETIC_P_GRID,
    seeds: list[int] = SYNTHETIC_SEEDS,
) -> pd.DataFrame:
    """Sweep (n, p, seed) over mutually independent columns and record W's null law.

    Validates A1's two empirical claims (paper Sec. 3 / App. C):
    (1) median W-hat scales as c / sqrt(n) -- record c_implied = w_median * sqrt(n)
        per cell; it should be roughly constant across n.
    (2) the induced quotient score rel/W has a heavy, finite-exponent upper tail
        (tail_slope, the Hill estimator already in this module).
    """
    rows: list[dict] = []
    total = len(n_grid) * len(p_grid) * len(seeds)
    done = 0
    for n in n_grid:
        for p in p_grid:
            for seed in seeds:
                X, y = _synthetic_independent_columns(n, p, seed)
                rm = redundancy_mass(X, y, k_selected=k_selected, random_state=seed)
                c_implied = rm.w_median * np.sqrt(n)
                rows.append(
                    {
                        "n": n,
                        "p": p,
                        "seed": seed,
                        "mass_below_abs": rm.mass_below_abs,
                        "mass_below_scaled": rm.mass_below_scaled,
                        "w_median": rm.w_median,
                        "w_p05": rm.w_p05,
                        "tail_slope": rm.tail_slope,
                        "c_implied": c_implied,
                    }
                )
                done += 1
                log(
                    f"  [{done}/{total}] n={n:>5} p={p:>5} seed={seed} "
                    f"w_median={rm.w_median:.5f} c_implied={c_implied:.4f} "
                    f"tail_slope={rm.tail_slope:.3f}"
                )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"wrote {len(df)} rows -> {out_csv}")

    c_hat = float(df["c_implied"].mean())
    c_std = float(df["c_implied"].std())
    c_cv = c_std / c_hat if c_hat else float("nan")
    finite_tail = df["tail_slope"].replace([np.inf, -np.inf], np.nan).dropna()
    alpha_hat = float(finite_tail.mean())
    log(
        f"SUMMARY: w_median ~ c/sqrt(n): c_hat={c_hat:.4f} "
        f"(std={c_std:.4f}, CV={c_cv:.4f}); mean Hill tail exponent alpha_hat={alpha_hat:.4f} "
        f"over {len(finite_tail)}/{len(df)} finite estimates"
    )
    return df


def run(out_csv: Path, out_fig: Path, k_selected: int, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(REAL_DATASETS) * len(seeds)
    done = 0
    for name in REAL_DATASETS:
        log(f"loading {name}")
        X, y, _task = load_dataset(name)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        for seed in seeds:
            rm = redundancy_mass(X, y, k_selected=k_selected, random_state=seed)
            rows.append(
                {
                    "dataset": name,
                    "p": X.shape[1],
                    "n": X.shape[0],
                    "seed": seed,
                    "mass_below_abs": rm.mass_below_abs,
                    "mass_below_scaled": rm.mass_below_scaled,
                    "w_median": rm.w_median,
                    "w_p05": rm.w_p05,
                    "tail_slope": rm.tail_slope,
                }
            )
            done += 1
            log(
                f"  [{done}/{total}] {name:11s} seed={seed} p={X.shape[1]:>5} "
                f"mass<={DANGER_THRESHOLD_ABS}(abs)={rm.mass_below_abs:.3f} "
                f"mass<=1/sqrt(n)(scaled)={rm.mass_below_scaled:.3f} median_W={rm.w_median:.3f}"
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"wrote {len(df)} rows -> {out_csv}")

    _plot(df, out_fig)
    return df


def _plot(df: pd.DataFrame, out_fig: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot the noise-floor-scaled statistic: it is comparable across datasets of
    # different n, unlike the fixed-threshold mass_below_abs (see module docstring).
    summary = df.groupby("dataset")["mass_below_scaled"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(summary.index, summary.to_numpy())
    ax.set_xlabel(r"fraction of candidates with $W \leq c/\sqrt{n}$ (noise-floor-scaled)")
    ax.set_title("Redundancy mass in the danger zone")
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    log(f"wrote {out_fig}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.redundancy_mass", description=__doc__)
    parser.add_argument("--out", type=str, default="results/redundancy_mass.csv")
    parser.add_argument("--fig", type=str, default="results/figures/redundancy_mass.pdf")
    parser.add_argument("--k-selected", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Run the synthetic arm instead: a sweep over mutually independent "
            "standard-normal columns across a grid of n and p, the construction "
            "that paper Sec. 3 / App. C cite as evidence for assumption A1. "
            "Writes results/redundancy_mass_synthetic.csv by default."
        ),
    )
    parser.add_argument(
        "--synthetic-out", type=str, default="results/redundancy_mass_synthetic.csv"
    )
    parser.add_argument("--synthetic-n", nargs="+", type=int, default=SYNTHETIC_N_GRID)
    parser.add_argument("--synthetic-p", nargs="+", type=int, default=SYNTHETIC_P_GRID)
    parser.add_argument("--synthetic-k-selected", type=int, default=SYNTHETIC_K_SELECTED)
    parser.add_argument("--synthetic-seeds", nargs="+", type=int, default=SYNTHETIC_SEEDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        run_synthetic(
            Path(args.synthetic_out),
            k_selected=args.synthetic_k_selected,
            n_grid=list(args.synthetic_n),
            p_grid=list(args.synthetic_p),
            seeds=list(args.synthetic_seeds),
        )
        return 0
    run(Path(args.out), Path(args.fig), args.k_selected, list(args.seeds))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
