"""M5 matched-settings operator box-and-whisker (fix-plan v3 F2.1).

Answers "is any operator better than the others?" directly: at the matched
measure pair ``dcor|pearson_abs|*|mean``, plots the per-dataset x per-seed
recovery F1 (panel a) and noise rate (panel b) distributions for the four
operator arms -- difference, quotient, multiplicative (the gate), and
reg_quotient -- over the golden suite (15 datasets x 10 seeds = 150 points
per arm), with a Friedman omnibus (datasets as blocks) plus Holm-corrected
pairwise Wilcoxon signed-rank tests.

Data provenance: ``difference``/``quotient``/``multiplicative`` rows already
exist per-run in ``results/factorial.parquet`` (the 10-seed factorial study).
``reg_quotient`` was never part of that grid (``mechanism/factorial.py``'s
``OPERATORS`` excludes it -- it was added later, only in the high-dimensional
p-sweep study). This module reruns just that one matched cell through the
identical golden-set harness (:func:`mechanism.factorial_protocol.run_factorial_grid`)
across the same 15 golden datasets, seeds, ks, and thresholds as the other
three arms, so the comparison is apples-to-apples; the result is cached to
``results/factorial_regquotient_golden.parquet`` so reruns are instant.

Standalone from ``analysis/paper_figures.py`` and ``analysis/paper_tables.py``
(owned by a different work package) per fix-plan F2.1 -- only shared,
read-only helpers are imported (``mechanism.paper_style``,
``analysis.display_names``).

CLI: ``uv run python -m analysis.operator_boxplot``.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from analysis.display_names import display_token
from analysis.paths import ARTIFACTS
from mechanism.datasets import list_mechanism_datasets
from mechanism.factorial import SelectorSpec
from mechanism.factorial_protocol import run_factorial_grid
from mechanism.paper_style import OKABE_ITO, paper_rc

FACTORIAL = Path("results/factorial.parquet")
REGQ_GOLDEN_CACHE = Path("results/factorial_regquotient_golden.parquet")
STATS_OUT = Path("results/stats_operator_matched.csv")
FIGURE_OUT = ARTIFACTS / "fig_operator_boxplot.pdf"

RELEVANCE_FAMILY = "dcor"  # spec token (mechanism.factorial.SelectorSpec)
RELEVANCE = "distance_corr"  # concrete scorer name in results/factorial.parquet
REDUNDANCY = "pearson_abs"
AGGREGATION = "mean"
ARMS = ["difference", "quotient", "multiplicative", "reg_quotient"]

# Same ks/thresholds/seeds mechanism.run_factorial used to produce the existing
# 3-arm golden rows (results/run_factorial_seeds10.log), so the reg_quotient
# rerun lands on identical (stop_mode, k, score_threshold) cells.
KS = [1, 2, 3, 5, 8, 10]
THRESHOLDS = [0.0, 0.05, 0.1, 0.2]
SEEDS = list(range(10))

ARM_COLORS = {
    "difference": OKABE_ITO["blue"],
    "quotient": OKABE_ITO["vermillion"],
    "multiplicative": OKABE_ITO["green"],
    # reg_quotient is the regularized quotient -- reuses the quotient hue.
    "reg_quotient": OKABE_ITO["vermillion"],
}


def _golden_dataset_names() -> list[str]:
    """The 15 golden (ground-truth) datasets already present in ``factorial.parquet``.

    ``mechanism.datasets.list_mechanism_datasets()`` has grown since the factorial
    study ran (26 datasets today), so we intersect with what's actually on disk
    rather than the full current registry -- keeps all four arms on the same
    15-dataset population.
    """
    df = pd.read_parquet(FACTORIAL, columns=["dataset"])
    golden_all = set(list_mechanism_datasets())
    names = sorted(set(df["dataset"].unique()) & golden_all)
    return names


def _load_or_run_reg_quotient(golden_datasets: list[str]) -> pd.DataFrame:
    """Load the cached golden-suite reg_quotient rows, or run the harness to produce them."""
    if REGQ_GOLDEN_CACHE.exists():
        print(f"[operator_boxplot] using cached reg_quotient rows: {REGQ_GOLDEN_CACHE}", flush=True)
        return pd.read_parquet(REGQ_GOLDEN_CACHE)

    spec = SelectorSpec(RELEVANCE_FAMILY, REDUNDANCY, "reg_quotient", AGGREGATION)
    total = len(golden_datasets)
    print(
        f"[operator_boxplot] no cached reg_quotient golden rows; running harness "
        f"({total} datasets x {len(SEEDS)} seeds, spec={spec.label!r})",
        flush=True,
    )
    frames: list[pd.DataFrame] = []
    t_start = time.perf_counter()
    for i, dataset in enumerate(golden_datasets, start=1):
        t0 = time.perf_counter()
        frame = run_factorial_grid(
            [spec], [dataset], KS, THRESHOLDS, SEEDS, include_downstream=False
        )
        frames.append(frame)
        elapsed = time.perf_counter() - t0
        total_elapsed = time.perf_counter() - t_start
        print(
            f"[operator_boxplot] [{i}/{total}] {dataset}: {len(frame)} rows "
            f"({elapsed:.1f}s, total {total_elapsed:.1f}s)",
            flush=True,
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        ["dataset", "spec", "stop_mode", "k", "score_threshold", "seed"], kind="stable"
    ).reset_index(drop=True)
    REGQ_GOLDEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REGQ_GOLDEN_CACHE, index=False)
    print(
        f"[operator_boxplot] wrote {len(df)} reg_quotient rows -> {REGQ_GOLDEN_CACHE}", flush=True
    )
    return df


def matched_per_run(refresh: bool = False) -> pd.DataFrame:
    """One row per (operator, dataset, seed) at ``dcor|pearson_abs|*|mean`` -- 150/arm.

    Collapses the (stop_mode, k, score_threshold) cells within each (operator,
    dataset, seed) into a mean, matching the unit the fix-plan spec calls for:
    "per-dataset x per-seed recovery F1 ... 15 datasets x 10 seeds = 150
    points/arm".
    """
    if refresh and REGQ_GOLDEN_CACHE.exists():
        REGQ_GOLDEN_CACHE.unlink()

    golden_datasets = _golden_dataset_names()
    if len(golden_datasets) != 15:
        raise ValueError(
            f"expected 15 golden datasets shared with factorial.parquet, found "
            f"{len(golden_datasets)}: {golden_datasets}"
        )

    df = pd.read_parquet(FACTORIAL)
    base = df[
        (df["relevance"] == RELEVANCE)
        & (df["redundancy"] == REDUNDANCY)
        & (df["aggregation"] == AGGREGATION)
        & (df["operator"].isin(["difference", "quotient", "multiplicative"]))
        & (df["dataset"].isin(golden_datasets))
        & df["f1"].notna()
        & df["error"].isna()
    ].copy()

    regq = _load_or_run_reg_quotient(golden_datasets)
    regq = regq[regq["f1"].notna() & regq["error"].isna()].copy()

    combined = pd.concat([base, regq], ignore_index=True)
    per_run = combined.groupby(["operator", "dataset", "seed"], as_index=False)[
        ["f1", "noise_rate"]
    ].mean()

    counts = per_run.groupby("operator").size()
    for arm in ARMS:
        n = int(counts.get(arm, 0))
        if n != len(golden_datasets) * len(SEEDS):
            print(
                f"[operator_boxplot] WARNING: arm {arm!r} has {n} (dataset, seed) points, "
                f"expected {len(golden_datasets) * len(SEEDS)}",
                flush=True,
            )
    return per_run


def _friedman_and_pairwise(per_run: pd.DataFrame, metric: str) -> list[dict]:
    """Friedman omnibus (n=15 dataset blocks) + Holm-corrected pairwise Wilcoxon.

    Both tests run on per-dataset means (averaged over seed and stop-config, matching
    the Friedman block unit) so the pairwise tests are paired on the same 15 blocks
    the omnibus used.
    """
    per_ds = per_run.groupby(["operator", "dataset"], as_index=False)[metric].mean()
    pivot = per_ds.pivot(index="dataset", columns="operator", values=metric)[ARMS]
    n_blocks = int(pivot.shape[0])

    stat, p = scipy_stats.friedmanchisquare(*[pivot[a] for a in ARMS])
    rows: list[dict] = [
        {
            "metric": metric,
            "test": "friedman",
            "arm_a": "",
            "arm_b": "",
            "statistic": float(stat),
            "p_raw": float(p),
            "p_holm": float("nan"),
            "n_blocks": n_blocks,
        }
    ]

    pairs = list(itertools.combinations(ARMS, 2))
    w_stats: list[float] = []
    raw_ps: list[float] = []
    for a, b in pairs:
        w_stat, w_p = scipy_stats.wilcoxon(pivot[a], pivot[b])
        w_stats.append(float(w_stat))
        raw_ps.append(float(w_p))
    _, holm_ps, _, _ = multipletests(raw_ps, method="holm")

    for (a, b), w_stat, p_raw, p_holm in zip(pairs, w_stats, raw_ps, holm_ps, strict=True):
        rows.append(
            {
                "metric": metric,
                "test": "wilcoxon_holm",
                "arm_a": a,
                "arm_b": b,
                "statistic": w_stat,
                "p_raw": p_raw,
                "p_holm": float(p_holm),
                "n_blocks": n_blocks,
            }
        )
    return rows


def compute_stats(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in ["f1", "noise_rate"]:
        rows.extend(_friedman_and_pairwise(per_run, metric))
    return pd.DataFrame(rows)


def fig_operator_boxplot(per_run: pd.DataFrame) -> Path:
    """Panel (a) recovery F1, panel (b) noise rate; one box+jittered-points per arm."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)  # deterministic jitter, matches paper_figures.py convention
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with paper_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
        for ax, metric, ylabel, title in zip(
            axes,
            ["f1", "noise_rate"],
            ["recovery F1", "noise rate"],
            ["(a) recovery F1", "(b) noise rate"],
            strict=True,
        ):
            for i, arm in enumerate(ARMS):
                vals = per_run.loc[per_run["operator"] == arm, metric].to_numpy()
                color = ARM_COLORS[arm]
                bp = ax.boxplot(
                    vals,
                    positions=[i],
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "lw": 1.2},
                    whiskerprops={"color": color, "lw": 0.8},
                    capprops={"color": color, "lw": 0.8},
                )
                for box in bp["boxes"]:
                    box.set(facecolor=color, alpha=0.35, edgecolor=color, lw=0.8)
                ax.plot(
                    i + rng.uniform(-0.14, 0.14, size=len(vals)),
                    vals,
                    "o",
                    color=color,
                    ms=2.5,
                    alpha=0.45,
                    mew=0,
                )
            ax.set_xticks(range(len(ARMS)))
            ax.set_xticklabels([display_token(a) for a in ARMS], rotation=20, ha="right")
            ax.set_xlim(-0.5, len(ARMS) - 0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        fig.savefig(FIGURE_OUT)
        plt.close(fig)
    return FIGURE_OUT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis.operator_boxplot", description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the cached reg_quotient golden-suite parquet and rerun the harness",
    )
    args = parser.parse_args(argv)

    per_run = matched_per_run(refresh=args.refresh)

    stats_df = compute_stats(per_run)
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(STATS_OUT, index=False)
    print(f"[operator_boxplot] wrote {STATS_OUT}", flush=True)

    fig_path = fig_operator_boxplot(per_run)
    print(f"[operator_boxplot] wrote {fig_path}", flush=True)

    summary = per_run.groupby("operator")[["f1", "noise_rate"]].describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    print(summary, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
