"""Mechanism-suite figure generators + summary table.

Consumes the tidy DataFrame produced by :func:`mechanism.protocol.run_mechanism_grid`
(columns == ``MECHANISM_COLUMNS``) and produces recovery figures plus a per-dataset,
per-measure-family summary table. Headless (Agg backend) so it runs under CI/pytest;
every plotting function saves to ``out_path``, closes its figure, and returns
``Path(out_path)``.

Also consumes the full-factorial DataFrame produced by
:func:`mechanism.factorial_protocol.run_factorial_grid` (columns ==
``FACTORIAL_COLUMNS``, with a ``spec`` column rather than ``method``): the
``operator_aggregation_heatmap``, ``ranking_leaderboard``,
``features_vs_threshold``, ``decision_guide_table``, and ``factorial_summary``
functions below operate on that schema. ``recovery_vs_k`` is generalised to
accept either schema (see its docstring).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mechanism.datasets import load_mechanism_dataset  # noqa: E402

_NONLINEAR_MARKERS = ("mutual_info", "distance", "dcor")
_LINEAR_MARKERS = ("pearson", "spearman")
_STOP_MODE_ORDER = ("fixed_k", "threshold", "val_fixed_k", "val_threshold")

_RECOVERY_METRIC_COLUMNS = (
    "precision",
    "recall",
    "f1",
    "redundancy_rate",
    "noise_rate",
    "average_precision",
    "roc_auc",
    "downstream_score",
)


def _classify_name(name: str) -> str:
    """Classify a single scorer name as ``"nonlinear"``, ``"linear"``, or ``"unknown"``."""
    if any(marker in name for marker in _NONLINEAR_MARKERS):
        return "nonlinear"
    if name == "" or name == "f" or name.startswith("f_"):
        return "linear"
    if any(marker in name for marker in _LINEAR_MARKERS):
        return "linear"
    return "unknown"


def measure_family(relevance: str, redundancy: str) -> str:
    """Classify a method's (relevance, redundancy) scorer pair into a measure family.

    Each name is first classified individually:
    - ``"nonlinear"`` if it contains ``"mutual_info"``, ``"distance"``, or ``"dcor"``
      (information-theoretic / distance-based measures that capture nonlinear
      dependence);
    - ``"linear"`` if it is empty, contains ``"pearson"`` or ``"spearman"``, or
      starts with ``"f_"``/equals ``"f"`` (the F-statistic filters, which are
      linear-model based);
    - ``"unknown"`` otherwise.

    The pair is then combined: ``"nonlinear"`` if BOTH names classify as
    nonlinear, ``"linear"`` if BOTH classify as linear, and ``"mixed"`` for any
    other combination (e.g. one linear + one nonlinear, or anything involving an
    unknown name).
    """
    relevance_family = _classify_name(relevance)
    redundancy_family = _classify_name(redundancy)
    if relevance_family == "nonlinear" and redundancy_family == "nonlinear":
        return "nonlinear"
    if relevance_family == "linear" and redundancy_family == "linear":
        return "linear"
    return "mixed"


def _with_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_family"] = [
        measure_family(rel, red)
        for rel, red in zip(out["relevance"].fillna(""), out["redundancy"].fillna(""), strict=True)
    ]
    return out


def _true_k(dataset: str) -> int:
    """Number of truly-informative features for ``dataset``, via its ``GroundTruth``."""
    _, _, _, gt = load_mechanism_dataset(dataset)
    return len(gt.informative)


def recovery_vs_k(df: pd.DataFrame, dataset: str, out_path: str | Path) -> Path:
    """Plot mean recovery F1 vs ``k``, one line per series group, for ``dataset``.

    Filters to ``dataset`` and ``stop_mode == "fixed_k"`` rows. Draws a vertical
    reference line at the dataset's true number of informative features
    (``true_k``, from its ``GroundTruth``) and annotates whether the F1 peak
    (of any series) coincides with ``true_k``.

    Accepts either DataFrame schema:
    - the factorial schema (``FACTORIAL_COLUMNS``, has a ``spec`` column): series
      are grouped by ``operator``;
    - the older mechanism schema (``MECHANISM_COLUMNS``, has a ``method`` column):
      series are grouped by measure family (``relevance``/``redundancy`` pair),
      exactly as before.
    """
    out_path = Path(out_path)
    subset = df[(df["dataset"] == dataset) & (df["stop_mode"] == "fixed_k")]

    if "spec" in df.columns:
        group_col = "operator"
        legend_title = "operator"
    else:
        subset = _with_family(subset)
        group_col = "_family"
        legend_title = "measure family"

    fig, ax = plt.subplots(figsize=(6, 4))
    peak_ks: list[float] = []
    for group in sorted(subset[group_col].dropna().unique()):
        group_df = subset[subset[group_col] == group]
        by_k = group_df.groupby("k")["f1"].mean().sort_index()
        ax.plot(by_k.index, by_k.to_numpy(), marker="o", label=str(group))
        if not by_k.empty:
            peak_ks.append(by_k.idxmax())

    true_k = _true_k(dataset)
    ax.axvline(true_k, color="red", linestyle="--", linewidth=1.0, label=f"true k={true_k}")
    peak_at_true_k = any(pk == true_k for pk in peak_ks)
    annotation = (
        f"F1 peak coincides with true k={true_k}"
        if peak_at_true_k
        else f"F1 peak does NOT coincide with true k={true_k}"
    )
    ax.text(
        0.02,
        0.02,
        annotation,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
    )

    ax.set_xlabel("k (selected features)")
    ax.set_ylabel("mean recovery F1")
    ax.set_title(f"Recovery F1 vs k — {dataset}")
    ax.legend(title=legend_title)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def linear_vs_nonlinear_gap(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Bar chart of mean F1(nonlinear family) - mean F1(linear family) per dataset.

    Uses ``stop_mode == "fixed_k"`` rows. Expected to be ~0 on ``linear_control_*``
    datasets and large/positive on nonlinear-mechanism datasets. A dataset missing
    one of the two families contributes a gap of 0.0 (skipped comparison).
    """
    out_path = Path(out_path)
    subset = _with_family(df[df["stop_mode"] == "fixed_k"])

    datasets = sorted(subset["dataset"].unique())
    gaps = []
    for dataset in datasets:
        d = subset[subset["dataset"] == dataset]
        nonlinear_f1 = d.loc[d["_family"] == "nonlinear", "f1"]
        linear_f1 = d.loc[d["_family"] == "linear", "f1"]
        if nonlinear_f1.empty or linear_f1.empty:
            gaps.append(0.0)
        else:
            gaps.append(float(nonlinear_f1.mean() - linear_f1.mean()))

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(datasets)), 4))
    ax.bar(range(len(datasets)), gaps)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right")
    ax.set_ylabel("mean F1 gap (nonlinear - linear)")
    ax.set_title("Nonlinear vs linear measure gap by dataset")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def fixed_k_vs_threshold(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Paired bar chart: best swept-k recovery F1 vs threshold-chosen recovery F1.

    Per dataset: "best swept-k" is the max over-k mean F1 among ``fixed_k`` rows;
    "threshold" is the mean F1 among ``threshold`` rows. Datasets missing one of
    the two stop modes are skipped.
    """
    out_path = Path(out_path)
    datasets = []
    best_fixed = []
    threshold_scores = []
    for dataset, d in df.groupby("dataset"):
        fixed = d[d["stop_mode"] == "fixed_k"]
        thresh = d[d["stop_mode"] == "threshold"]
        if fixed.empty or thresh.empty:
            continue
        best_fixed_f1 = fixed.groupby("k")["f1"].mean().max()
        threshold_f1 = thresh["f1"].mean()
        datasets.append(dataset)
        best_fixed.append(float(best_fixed_f1))
        threshold_scores.append(float(threshold_f1))

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(datasets)), 4))
    x = range(len(datasets))
    width = 0.35
    ax.bar([i - width / 2 for i in x], best_fixed, width, label="best swept-k")
    ax.bar([i + width / 2 for i in x], threshold_scores, width, label="threshold")
    ax.set_xticks(list(x), datasets, rotation=30, ha="right")
    ax.set_ylabel("recovery F1")
    ax.set_title("Best fixed-k vs threshold-chosen recovery")
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def mechanism_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, measure family) mean recovery + downstream metrics.

    Means pool across ``stop_mode`` and every ``k`` (unlike the figure
    functions, which filter/split on ``stop_mode``): each row is a single
    mean over all fixed_k and threshold rows for that (dataset, family).
    NaN ``downstream_score`` values are skipped (pandas ``skipna=True``).

    Returns a tidy DataFrame with columns
    ``["dataset", "family", "precision", "recall", "f1", "redundancy_rate",
    "noise_rate", "downstream_score"]``.
    """
    subset = _with_family(df)
    grouped = (
        subset.groupby(["dataset", "_family"])[
            ["precision", "recall", "f1", "redundancy_rate", "noise_rate", "downstream_score"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"_family": "family"})
    )
    return grouped


def operator_aggregation_heatmap(df: pd.DataFrame, metric: str, out_path: str | Path) -> Path:
    """Heatmap of operator x aggregation mean ``metric``, one panel per relevance measure.

    Consumes the factorial schema (``operator``, ``aggregation``, ``relevance``
    columns). One panel is drawn per distinct ``relevance`` value; within a panel,
    rows are operators, columns are aggregations, and each cell is annotated with
    its mean ``metric`` value. The ``(operator="multiplicative",
    aggregation="max")`` cell -- the ModMRMR niche -- is marked with a trailing
    ``"*"`` wherever it appears, so the niche is visually findable across panels.
    """
    out_path = Path(out_path)
    relevances = sorted(df["relevance"].dropna().unique())
    n_panels = max(len(relevances), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4), squeeze=False)
    axes = axes[0]
    for ax, relevance in zip(axes, relevances, strict=True):
        sub = df[df["relevance"] == relevance]
        pivot = sub.pivot_table(
            index="operator", columns="aggregation", values=metric, aggfunc="mean"
        )
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), list(pivot.columns), rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)), list(pivot.index))
        ax.set_title(relevance, fontsize=10)
        for i, operator in enumerate(pivot.index):
            for j, aggregation in enumerate(pivot.columns):
                val = pivot.iloc[i, j]
                if pd.isna(val):
                    continue
                label = f"{val:.2f}"
                if operator == "multiplicative" and aggregation == "max":
                    label += "*"  # the ModMRMR niche cell
                ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"operator x aggregation — mean {metric} (* = ModMRMR niche cell)")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def ranking_leaderboard(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Threshold-free bar chart: mean ``average_precision`` per ``spec``, ranked descending."""
    out_path = Path(out_path)
    means = df.groupby("spec")["average_precision"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(means)), 4))
    ax.bar(range(len(means)), means.to_numpy())
    ax.set_xticks(range(len(means)), list(means.index), rotation=45, ha="right")
    ax.set_ylabel("mean average precision")
    ax.set_title("Ranking leaderboard (mean average precision by spec)")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def _features_vs_threshold_summary(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Per stop-mode mean recovery F1 and mean selected size ``k`` for ``dataset``.

    Only stop modes actually present in ``df`` for ``dataset`` are included, in
    the fixed canonical order ``_STOP_MODE_ORDER``. Pure computation, factored out
    of :func:`features_vs_threshold` so the "covers every present stop mode"
    invariant is directly testable without inspecting a rendered figure.
    """
    subset = df[df["dataset"] == dataset]
    present = set(subset["stop_mode"].unique())
    rows = []
    for mode in _STOP_MODE_ORDER:
        if mode not in present:
            continue
        mode_df = subset[subset["stop_mode"] == mode]
        rows.append(
            {
                "stop_mode": mode,
                "mean_f1": float(mode_df["f1"].mean()),
                "mean_k": float(mode_df["k"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=["stop_mode", "mean_f1", "mean_k"])


def features_vs_threshold(df: pd.DataFrame, dataset: str, out_path: str | Path) -> Path:
    """The #features-vs-threshold comparison: recovery F1 and selected size per stop mode.

    For ``dataset``, for EVERY stop mode present in ``df``
    (``fixed_k``/``threshold``/``val_fixed_k``/``val_threshold``), plots the mean
    recovery F1 (left panel) and the mean selected feature-set size ``k`` (right
    panel), with a reference line at the dataset's true number of informative
    features (``true_k``) on the size panel -- so a reader can see where the
    fixed-k sweep's best k, the threshold-chosen size, and the
    validation-selected size each land relative to ``true_k``.

    Generalises the committed :func:`fixed_k_vs_threshold` (which only compared
    the ``fixed_k``/``threshold`` pair on recovery F1) to all four stop modes and
    to feature-set size.
    """
    out_path = Path(out_path)
    summary = _features_vs_threshold_summary(df, dataset)
    true_k = _true_k(dataset)

    fig, (ax_f1, ax_k) = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(summary))
    labels = list(summary["stop_mode"])

    ax_f1.bar(x, summary["mean_f1"].to_numpy())
    ax_f1.set_xticks(list(x), labels, rotation=30, ha="right")
    ax_f1.set_ylabel("mean recovery F1")
    ax_f1.set_title(f"Recovery F1 by stop mode — {dataset}")

    ax_k.bar(x, summary["mean_k"].to_numpy())
    ax_k.axhline(true_k, color="red", linestyle="--", linewidth=1.0, label=f"true k={true_k}")
    ax_k.set_xticks(list(x), labels, rotation=30, ha="right")
    ax_k.set_ylabel("mean selected size k")
    ax_k.set_title(f"# selected features by stop mode — {dataset}")
    ax_k.legend()

    fig.suptitle(f"#features vs threshold — {dataset}")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def decision_guide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Practitioner recommendation table: best ``spec`` per dataset by F1 and by noise_rate.

    Per ``dataset``, reports the ``spec`` with the highest mean ``f1`` and the
    ``spec`` with the lowest (best) mean ``noise_rate`` -- pooling across every
    ``stop_mode``, ``k``, and ``seed`` row for that dataset. Returns a tidy
    DataFrame with columns ``["dataset", "dependence", "best_spec_by_f1",
    "best_f1", "best_spec_by_noise_rate", "best_noise_rate"]``.
    """
    rows = []
    for dataset, d in df.groupby("dataset"):
        f1_by_spec = d.groupby("spec")["f1"].mean()
        noise_by_spec = d.groupby("spec")["noise_rate"].mean()
        best_f1_spec = f1_by_spec.idxmax()
        best_noise_spec = noise_by_spec.idxmin()
        rows.append(
            {
                "dataset": dataset,
                "dependence": d["dependence"].iloc[0],
                "best_spec_by_f1": best_f1_spec,
                "best_f1": float(f1_by_spec.loc[best_f1_spec]),
                "best_spec_by_noise_rate": best_noise_spec,
                "best_noise_rate": float(noise_by_spec.loc[best_noise_spec]),
            }
        )
    return pd.DataFrame(rows)


def factorial_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (spec, dependence) mean of every recovery + ranking + downstream metric.

    Pools across ``dataset`` (within a dependence class), ``stop_mode``, ``k``,
    and ``seed``. Returns a tidy DataFrame with columns ``["spec", "dependence",
    "precision", "recall", "f1", "redundancy_rate", "noise_rate",
    "average_precision", "roc_auc", "downstream_score"]``.
    """
    grouped = (
        df.groupby(["spec", "dependence"])[list(_RECOVERY_METRIC_COLUMNS)].mean().reset_index()
    )
    return grouped


_MI_INCUMBENT = "mi_reg_k3"
_MI_REGRESSION_MARGIN = 0.02


def mi_ranking_leaderboard(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Threshold-free bar chart: mean ``average_precision`` per ``estimator``, ranked descending.

    Consumes the MI-comparison schema (``MI_COMPARISON_COLUMNS``, from
    :func:`mechanism.mi_comparison.run_mi_comparison_grid`). Rows with NaN
    ``average_precision`` (error cells) are dropped before aggregating. Each bar
    is labelled with its mean value.
    """
    out_path = Path(out_path)
    subset = df.dropna(subset=["average_precision"])
    means = subset.groupby("estimator")["average_precision"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(means)), 4))
    bars = ax.bar(range(len(means)), means.to_numpy())
    ax.set_xticks(range(len(means)), list(means.index), rotation=45, ha="right")
    ax.set_ylabel("mean average precision")
    ax.set_title("MI-estimator ranking leaderboard (mean average precision)")
    for bar, value in zip(bars, means.to_numpy(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def mi_ap_by_dependence(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Grouped bar chart: mean ``average_precision`` per ``estimator`` x ``dependence`` class.

    Consumes the MI-comparison schema. Rows with NaN ``average_precision``
    (error cells) are dropped before aggregating. Exposes whether any MI
    estimator regresses on nonlinear-dependence datasets relative to
    linear/mixed ones.
    """
    out_path = Path(out_path)
    subset = df.dropna(subset=["average_precision"])
    pivot = subset.pivot_table(
        index="estimator", columns="dependence", values="average_precision", aggfunc="mean"
    )
    estimators = list(pivot.index)
    dependences = sorted(pivot.columns)
    n_dep = max(len(dependences), 1)
    width = 0.8 / n_dep

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(estimators)), 4))
    x = range(len(estimators))
    for i, dependence in enumerate(dependences):
        values = pivot[dependence].to_numpy(dtype=float)
        offsets = [xi + (i - (n_dep - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width, label=dependence)
    ax.set_xticks(list(x), estimators, rotation=45, ha="right")
    ax.set_ylabel("mean average precision")
    ax.set_title("MI-estimator average precision by dependence class")
    ax.legend(title="dependence")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def mi_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (estimator, dependence) mean of every recovery + ranking + downstream metric.

    Pools across ``dataset``, ``k``, and ``seed`` (NaN-skipping mean, so error
    rows with NaN metrics do not distort the average). Returns a tidy DataFrame
    with columns ``["estimator", "dependence", "precision", "recall", "f1",
    "redundancy_rate", "noise_rate", "average_precision", "roc_auc",
    "downstream_score"]``.
    """
    grouped = (
        df.groupby(["estimator", "dependence"])[list(_RECOVERY_METRIC_COLUMNS)].mean().reset_index()
    )
    return grouped


def mi_winner(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``estimator``, ranked by golden mean average precision, descending.

    ``mean_ap_golden`` is the mean ``average_precision`` over every (non-error)
    row for the estimator, regardless of ``dependence`` class.
    ``mean_ap_nonlinear``/``mean_ap_linear`` restrict that mean to
    ``dependence == "nonlinear"``/``"linear"`` rows (NaN if the estimator has
    none). ``regresses_nonlinear`` is ``True`` iff the estimator's
    ``mean_ap_nonlinear`` falls more than 0.02 below the incumbent
    (``"mi_reg_k3"``) ``mean_ap_nonlinear``. If ``mi_reg_k3`` is absent from
    ``df``, or either side of the comparison is NaN, ``regresses_nonlinear`` is
    ``False`` (never NaN, never raises). The top-ranked row with
    ``regresses_nonlinear == False`` is the winner; the caller selects it.
    """
    subset = df.dropna(subset=["average_precision"])

    has_incumbent = _MI_INCUMBENT in set(df["estimator"])
    incumbent_rows = subset[subset["estimator"] == _MI_INCUMBENT]
    incumbent_nonlinear = incumbent_rows.loc[
        incumbent_rows["dependence"] == "nonlinear", "average_precision"
    ]
    incumbent_nonlinear_ap = (
        float(incumbent_nonlinear.mean()) if not incumbent_nonlinear.empty else float("nan")
    )

    rows = []
    for estimator, group in subset.groupby("estimator"):
        mean_ap_golden = float(group["average_precision"].mean())
        nonlinear_ap = group.loc[group["dependence"] == "nonlinear", "average_precision"]
        linear_ap = group.loc[group["dependence"] == "linear", "average_precision"]
        mean_ap_nonlinear = float(nonlinear_ap.mean()) if not nonlinear_ap.empty else float("nan")
        mean_ap_linear = float(linear_ap.mean()) if not linear_ap.empty else float("nan")

        if not has_incumbent or pd.isna(mean_ap_nonlinear) or pd.isna(incumbent_nonlinear_ap):
            regresses_nonlinear = False
        else:
            regresses_nonlinear = mean_ap_nonlinear < (
                incumbent_nonlinear_ap - _MI_REGRESSION_MARGIN
            )

        rows.append(
            {
                "estimator": estimator,
                "mean_ap_golden": mean_ap_golden,
                "mean_ap_nonlinear": mean_ap_nonlinear,
                "mean_ap_linear": mean_ap_linear,
                "regresses_nonlinear": bool(regresses_nonlinear),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "estimator",
            "mean_ap_golden",
            "mean_ap_nonlinear",
            "mean_ap_linear",
            "regresses_nonlinear",
        ],
    )
    return result.sort_values("mean_ap_golden", ascending=False).reset_index(drop=True)
