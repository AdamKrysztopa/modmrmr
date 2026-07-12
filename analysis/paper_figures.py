"""Publication figures for the OpenMRMR paper (design spec §4).

Each figure has a pure prep function (DataFrame in, tidy DataFrame with
bootstrap CIs out — tested) and a thin plot function writing a vector PDF
into PAPER_DIR/artifacts/ (see analysis.paths). CLI: `uv run python -m analysis.paper_figures`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from analysis.display_names import display_token
from analysis.paths import ARTIFACTS
from mechanism.paper_style import (
    MEASURE_COLORS,
    OKABE_ITO,
    OPERATOR_COLORS,
    bootstrap_ci,
    paper_rc,
)

FACTORIAL = Path("results/factorial.parquet")
HIGHDIM = Path("results/highdim_study.parquet")
HIGHDIM_DOWNSTREAM_SYNTH = Path("results/highdim_downstream_synth.csv")

# operator/aggregation pairs shown in operator figures, keyed by OPERATOR_COLORS names
VARIANTS = {
    ("difference", "mean"): "difference",
    ("multiplicative", "mean"): "multiplicative",
    ("quotient", "mean"): "quotient",
    ("reg_quotient", "mean"): "reg_quotient",
    ("multiplicative", "max"): "mult_max",
}
VARIANT_LABELS = {
    "difference": "difference (mean)",
    "multiplicative": "gate (mean)",
    "quotient": "quotient (mean)",
    "reg_quotient": "reg. quotient (mean)",
    "mult_max": "gate (max)",
}

# --- Money figure (Fig. of Sec. 5) styling: the four operator arms, directly
# labeled so no legend floats over the data (reviewer B-6). reg_quotient reuses
# the quotient hue, dashed, to read as "the quotient, regularized".
MONEY_ORDER = ["quotient", "reg_quotient", "difference", "multiplicative"]
MONEY_COLORS = {**OPERATOR_COLORS, "reg_quotient": OPERATOR_COLORS["quotient"]}
MONEY_LINESTYLE = {"reg_quotient": (0, (4, 2))}
MONEY_LABEL = {
    "quotient": r"quotient $D/W$",
    "reg_quotient": r"reg. quotient $D/(W{+}\epsilon)$",
    "difference": r"difference $D{-}W$",
    "multiplicative": r"gate $D(1{-}W)$",
}


def ci_agg(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Group and reduce to mean + seed-deterministic 95% bootstrap CI."""
    rows = []
    for i, (key, g) in enumerate(df.groupby(group_cols, sort=True)):
        key = key if isinstance(key, tuple) else (key,)
        lo, hi = bootstrap_ci(g[value_col].to_numpy(), seed=i)
        rows.append(
            dict(zip(group_cols, key, strict=True))
            | {"mean": float(g[value_col].mean()), "ci_lo": lo, "ci_hi": hi}
        )
    return pd.DataFrame(rows)


def _variant_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["variant"] = [
        VARIANTS.get((op, agg)) for op, agg in zip(out["operator"], out["aggregation"], strict=True)
    ]
    return out[out["variant"].notna()]


def prep_quotient_collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic p-sweep at k=20: recall / AP / noise per (variant, p) with CIs."""
    syn = _variant_frame(df[(df["kind"] == "synthetic") & (df["k"] == 20)])
    metrics = ["recall_informative", "average_precision", "noise_rate"]
    long = syn.melt(
        id_vars=["variant", "p", "seed"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    return ci_agg(long, ["variant", "p", "metric"], "value")


def prep_highdim_real(df: pd.DataFrame) -> pd.DataFrame:
    """Real high-dim downstream score at k=20 per (dataset, variant) with CIs."""
    real = _variant_frame(df[(df["kind"] == "real") & (df["k"] == 20)])
    return ci_agg(real, ["dataset", "variant"], "downstream_score")


METRIC_TITLES = {
    "recall_informative": "Recall of the 5 informative features",
    "average_precision": "Average precision (ranking)",
    "noise_rate": "Noise rate of the selection",
}


def fig_quotient_collapse() -> Path:
    """Money figure: noise rate and recall vs dimension p, four operator arms.

    Two panels, directly labeled (no legend over the data). The default
    unregularized quotient's noise climbs monotonically with p while the bounded
    gate stays flat; a regularized quotient D/(W+eps) (dashed) removes most of the
    collapse too, isolating the *unbounded denominator* as the cause.
    """
    import matplotlib.pyplot as plt

    df = pd.read_parquet(HIGHDIM)
    sweep = prep_quotient_collapse(df)
    pmax = sweep["p"].max()

    def _val(variant: str, metric: str, p: float) -> float:
        row = sweep[(sweep["variant"] == variant) & (sweep["metric"] == metric) & (sweep["p"] == p)]
        return float(row["mean"].iloc[0]) if len(row) else float("nan")

    panels = [
        ("noise_rate", "noise rate of selection"),
        ("recall_informative", "recall of 5 informative"),
    ]
    with paper_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
        for ax, (metric, ylabel) in zip(axes, panels, strict=True):
            for variant in MONEY_ORDER:
                sub = sweep[(sweep["variant"] == variant) & (sweep["metric"] == metric)]
                sub = sub.sort_values("p")
                if sub.empty:
                    continue
                color = MONEY_COLORS[variant]
                ls = MONEY_LINESTYLE.get(variant, "-")
                ax.plot(sub["p"], sub["mean"], marker="o", ms=2.5, lw=1.4, color=color, ls=ls)
                ax.fill_between(sub["p"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.10)
                # On the noise panel the four arms separate cleanly, so label each
                # directly. On the recall panel gate/difference/reg-quotient all sit
                # at ~0.68 (they coincide) and only the quotient is apart, so label
                # just the quotient there and group the rest with one note (below).
                label_here = metric == "noise_rate" or variant == "quotient"
                if label_here:
                    last = sub.iloc[-1]
                    ax.annotate(
                        MONEY_LABEL[variant],
                        (last["p"], last["mean"]),
                        textcoords="offset points",
                        xytext=(5, 0),
                        va="center",
                        fontsize=6.2,
                        color=color,
                    )
            ax.set_xscale("log")
            ax.set_xlabel("number of features $p$")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
            ax.set_xlim(sweep["p"].min() * 0.85, pmax * 4.5)
        # recall panel: one grouped label for the three coincident high arms
        axes[1].annotate(
            "gate, difference,\nreg. quotient",
            xy=(pmax, 0.68),
            textcoords="offset points",
            xytext=(5, 6),
            va="center",
            fontsize=6.0,
            color="black",
        )
        # ratio annotation on the noise panel (computed, never hard-coded). Placed
        # in the empty band below the gate line with a short arrow up to the gate's
        # left end, so it never crosses the difference / reg-quotient lines.
        pmin = sweep["p"].min()
        q_noise = _val("quotient", "noise_rate", pmax)
        g_noise = _val("multiplicative", "noise_rate", pmax)
        g_lo = _val("multiplicative", "noise_rate", pmin)
        if q_noise == q_noise and g_noise and g_noise == g_noise:
            axes[0].annotate(
                f"gate: {q_noise / g_noise:.1f}$\\times$ less noise\nthan the quotient",
                xy=(pmin, g_lo),
                xytext=(pmin * 1.05, 0.08),
                fontsize=6.2,
                va="center",
                ha="left",
                arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.35"},
            )
        out = ARTIFACTS / "fig3_quotient_collapse.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def fig_highdim_real() -> Path:
    """Appendix companion: real high-dim downstream RF score by operator variant,
    as box-and-whisker over the four real datasets x 3 seeds (k=20). The datasets
    sit at different score baselines, so the spread is wide by construction."""
    import matplotlib.pyplot as plt
    import numpy as np

    real = _variant_frame(pd.read_parquet(HIGHDIM).query("kind == 'real' and k == 20"))
    variants = ["difference", "multiplicative", "quotient", "mult_max"]
    rng = np.random.default_rng(0)
    with paper_rc():
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        groups = [
            (
                v,
                OPERATOR_COLORS[v],
                real.loc[real["variant"] == v, "downstream_score"].dropna().to_numpy(),
            )
            for v in variants
        ]
        _boxplot_by_group(ax, groups, rng)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=15, fontsize=6)
        ax.set_xlim(-0.5, len(variants) - 0.5)
        ax.set_ylabel("downstream RF score (real high-dim)")
        out = ARTIFACTS / "fig_highdim_real.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def prep_highdim_downstream_synth(df: pd.DataFrame) -> pd.DataFrame:
    """Downstream RF balanced accuracy by operator x p (synthetic p-sweep, k=20),
    10 seeds, with seed-deterministic bootstrap CIs — see
    ``analysis/highdim_downstream_synth.py``."""
    return ci_agg(df, ["operator", "p"], "downstream_score")


def fig_highdim_downstream() -> Path:
    """Companion figure: downstream (held-out RF) balanced accuracy vs dimension p
    for the four operator arms on the synthetic p-sweep (k=20, 10 seeds). Unlike
    the recovery-metric money figure (fig3_quotient_collapse, untouched here), this
    shows the raw quotient's predictive cost widening with p while difference,
    multiplicative, and reg_quotient hold steady — directly labeled, no legend."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(HIGHDIM_DOWNSTREAM_SYNTH)
    prep = prep_highdim_downstream_synth(df)
    pmax = prep["p"].max()
    pmin = prep["p"].min()

    with paper_rc():
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for variant in MONEY_ORDER:
            sub = prep[prep["operator"] == variant].sort_values("p")
            if sub.empty:
                continue
            color = MONEY_COLORS[variant]
            ls = MONEY_LINESTYLE.get(variant, "-")
            ax.plot(sub["p"], sub["mean"], marker="o", ms=2.5, lw=1.4, color=color, ls=ls)
            ax.fill_between(sub["p"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.12)
            last = sub.iloc[-1]
            ax.annotate(
                MONEY_LABEL[variant],
                (last["p"], last["mean"]),
                textcoords="offset points",
                xytext=(5, 0),
                va="center",
                fontsize=6.2,
                color=color,
            )
        ax.set_xscale("log")
        ax.set_xlabel("number of features $p$")
        ax.set_ylabel("downstream RF balanced accuracy")
        ax.set_xlim(pmin * 0.85, pmax * 4.5)
        out = ARTIFACTS / "fig_highdim_downstream.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


RELEVANCE_ORDER = [
    "pearson_abs",
    "spearman_abs",
    "f_classif",
    "f_regression",
    "mutual_info_classif",
    "mutual_info_regression",
    "distance_corr",
]
RELEVANCE_FAMILY_COLOR = {  # MEASURE_COLORS keys from paper_style
    "pearson_abs": "linear",
    "spearman_abs": "linear",
    "f_classif": "f_test",
    "f_regression": "f_test",
    "mutual_info_classif": "mutual_info",
    "mutual_info_regression": "mutual_info",
    "distance_corr": "distance_corr",
}

DEPENDENCE_PANELS = ["linear", "nonlinear", "mixed"]


def golden(df: pd.DataFrame) -> pd.DataFrame:
    """Golden-set rows (recovery ground truth available): f1 scored, no error."""
    return df[df["f1"].notna() & df["error"].isna()]


def prep_measure_dependence(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recovery F1 per relevance measure x dependence class.

    Collapses to per-(relevance, dependence, dataset) means first, then
    bootstraps the CI over datasets — the dataset is the unit of evidence.
    """
    g = golden(df)
    per_ds = g.groupby(["relevance", "dependence", "dataset"], as_index=False)["f1"].mean()
    return ci_agg(per_ds, ["relevance", "dependence"], "f1")


def prep_measure_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(relevance, dependence, dataset) mean recovery F1 — one row per point."""
    g = golden(df)
    return g.groupby(["relevance", "dependence", "dataset"], as_index=False)["f1"].mean()


def fig_measure_dependence() -> Path:
    """Fig 2: Rule-1 crossover as box-and-whisker — measures on a fixed y-order,
    two panels (linear vs nonlinear); the high boxes move from the top group
    (linear measures) to the bottom group (MI / distance correlation)."""
    import matplotlib.pyplot as plt
    import numpy as np

    per_ds = prep_measure_boxes(pd.read_parquet(FACTORIAL))
    order = list(reversed(RELEVANCE_ORDER))  # top of axis = first in RELEVANCE_ORDER
    ypos = {rel: i for i, rel in enumerate(order)}
    rng = np.random.default_rng(0)  # deterministic jitter for the overlaid points
    with paper_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True)
        for ax, dep, title in zip(
            axes, ["linear", "nonlinear"], ["linear structure", "nonlinear structure"], strict=True
        ):
            for rel in order:
                vals = per_ds[(per_ds["relevance"] == rel) & (per_ds["dependence"] == dep)][
                    "f1"
                ].to_numpy()
                if len(vals) == 0:
                    continue
                color = MEASURE_COLORS[RELEVANCE_FAMILY_COLOR[rel]]
                y = ypos[rel]
                bp = ax.boxplot(
                    vals,
                    positions=[y],
                    vert=False,
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "lw": 1.0},
                    whiskerprops={"color": color, "lw": 0.8},
                    capprops={"color": color, "lw": 0.8},
                )
                for box in bp["boxes"]:
                    box.set(facecolor=color, alpha=0.35, edgecolor=color, lw=0.8)
                # overlay the individual dataset points (n is small — show them)
                ax.plot(
                    vals,
                    y + rng.uniform(-0.16, 0.16, size=len(vals)),
                    "o",
                    ms=2.5,
                    color=color,
                    alpha=0.9,
                )
            ax.set_title(title)
            ax.set_xlim(0.0, 0.85)
            ax.set_xlabel("recovery F1")
            ax.axvline(0.0, color="none")
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels(order)
        axes[0].set_ylim(-0.6, len(order) - 0.4)
        out = ARTIFACTS / "fig2_measure_dependence.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def prep_operator_lowdim(df: pd.DataFrame) -> pd.DataFrame:
    """Golden-set operator means for f1 and noise_rate, CI over datasets."""
    g = golden(df)
    per_ds = g.groupby(["operator", "dataset"], as_index=False)[["f1", "noise_rate"]].mean()
    long = per_ds.melt(
        id_vars=["operator", "dataset"],
        value_vars=["f1", "noise_rate"],
        var_name="metric",
        value_name="value",
    )
    return ci_agg(long, ["operator", "metric"], "value")


def prep_operator_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Operator x aggregation cell means (golden) for the twin heatmaps."""
    g = golden(df)
    cells = g.groupby(["operator", "aggregation"], as_index=False)[["f1", "noise_rate"]].mean()
    return cells.melt(id_vars=["operator", "aggregation"], var_name="metric", value_name="mean")


def fig_operator_lowdim() -> Path:
    """Fig 4: honest low-dim operator comparison — F1s overlap, noise doesn't."""
    import matplotlib.pyplot as plt

    df = pd.read_parquet(FACTORIAL)
    prep = prep_operator_lowdim(df)
    f1 = prep[prep["metric"] == "f1"].set_index("operator")
    noise = prep[prep["metric"] == "noise_rate"].set_index("operator")
    with paper_rc():
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        for operator in ["difference", "multiplicative", "quotient"]:
            x, xlo, xhi = f1.loc[operator, ["mean", "ci_lo", "ci_hi"]]
            y, ylo, yhi = noise.loc[operator, ["mean", "ci_lo", "ci_hi"]]
            color = OPERATOR_COLORS[operator]
            ax.plot(
                [xlo, xhi],
                [y, y],
                color=color,
                lw=1.0,
            )
            ax.plot(
                [x, x],
                [ylo, yhi],
                color=color,
                lw=1.0,
            )
            ax.plot(x, y, "o", color=color, ms=5)
            ax.annotate(
                display_token(operator),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=color,
            )
        ax.set_xlim(0.35, 0.55)
        ax.set_ylim(0.2, 0.5)
        ax.set_xlabel("mean recovery F1")
        ax.set_ylabel("mean noise rate")
        out = ARTIFACTS / "fig4_operator_lowdim.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def fig_operator_aggregation() -> Path:
    """Fig 5: operator x aggregation heatmaps for F1 and noise_rate."""
    import matplotlib.pyplot as plt

    df = pd.read_parquet(FACTORIAL)
    prep = prep_operator_aggregation(df)
    operators = ["difference", "multiplicative", "quotient"]
    aggregations = ["mean", "max", "sum"]
    with paper_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
        # F1: higher is better (viridis); noise: higher is worse (reversed) so the
        # two panels never share a scale across their opposite semantics. Each panel
        # gets its own vmin/vmax fit to its data, so cells are not clipped.
        for ax, metric, title, cmap in zip(
            axes,
            ["f1", "noise_rate"],
            ["mean recovery F1 (higher better)", "mean noise rate (lower better)"],
            ["viridis", "magma_r"],
            strict=True,
        ):
            sub = prep[prep["metric"] == metric]
            grid = sub.pivot(index="aggregation", columns="operator", values="mean").reindex(
                index=aggregations, columns=operators
            )
            arr = grid.to_numpy()
            vmin, vmax = float(arr.min()), float(arr.max())
            cmap_obj = plt.get_cmap(cmap)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            im = ax.imshow(arr, cmap=cmap_obj, norm=norm, aspect="auto")
            ax.set_xticks(range(len(operators)))
            ax.set_xticklabels([display_token(op) for op in operators], rotation=20)
            ax.set_yticks(range(len(aggregations)))
            ax.set_yticklabels(aggregations)
            ax.set_title(title)
            for i in range(len(aggregations)):
                for j in range(len(operators)):
                    value = grid.iloc[i, j]
                    # Pick text color from the CELL's background luminance, not the
                    # value: for the reversed noise colormap low value = light bg, so
                    # a value-based rule would invert (white text on light cells).
                    r, g, b, _ = cmap_obj(norm(value))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="black" if lum > 0.55 else "white",
                        fontsize=7,
                    )
            # outline the multiplicative+mean cell — the gate configuration this
            # paper recommends (mean, not max; see Sec. 5).
            i = aggregations.index("mean")
            j = operators.index("multiplicative")
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.5)
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = ARTIFACTS / "fig5_operator_aggregation.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def prep_stopping(df: pd.DataFrame) -> pd.DataFrame:
    """Recovery F1 and mean selected size per stopping mode (golden sets)."""
    g = golden(df)
    per_ds = g.groupby(["stop_mode", "dataset"], as_index=False)[["f1", "k"]].mean()
    long = per_ds.melt(
        id_vars=["stop_mode", "dataset"],
        value_vars=["f1", "k"],
        var_name="metric",
        value_name="value",
    )
    return ci_agg(long, ["stop_mode", "metric"], "value")


MI_WINNER_CSV = Path("results/mi_comparison_winner.csv")
# Study-4 gate reference: pearson_abs mean ranking-AP on linear sets, as
# recorded in MI_WINNER_CSV above.
PEARSON_LINEAR_AP = 0.731


def prep_mi_wash() -> pd.DataFrame:
    """MI estimator wash: 12 estimators, sorted by golden AP descending."""
    df = pd.read_csv(MI_WINNER_CSV)
    cols = ["estimator", "mean_ap_linear", "mean_ap_nonlinear", "mean_ap_golden"]
    return df[cols].sort_values("mean_ap_golden", ascending=False).reset_index(drop=True)


def fig_mi_wash() -> Path:
    """Fig 6: MI-estimator wash — linear vs nonlinear AP with Pearson baseline.

    Compact (~60% text-width). The rows are ordered by golden AP descending; that
    sort key is made explicit with a faint grey tick per row and a note (reviewer
    B-10). The two named outliers — the Gaussian-copula ``gcmi`` and ``bspline_mi``,
    both of which collapse on nonlinear structure — are labeled directly.
    """
    import matplotlib.pyplot as plt

    prep = prep_mi_wash()

    with paper_rc():
        fig, ax = plt.subplots(figsize=(3.7, 3.9))
        y = range(len(prep))

        # Faint grey tick at the golden AP per row — the visible sort key. Because the
        # rows are sorted by this column, the ticks descend monotonically down the axis.
        ax.plot(
            prep["mean_ap_golden"],
            y,
            marker="|",
            color="0.55",
            ms=9,
            mew=1.1,
            ls="",
            label="golden AP (sort key)",
        )

        # Two labeled dot series per estimator row: linear (blue), nonlinear (orange)
        ax.plot(
            prep["mean_ap_linear"],
            y,
            "o",
            color=MEASURE_COLORS["linear"],
            ms=4,
            ls="",
            label="linear sets",
        )
        ax.plot(
            prep["mean_ap_nonlinear"],
            y,
            "o",
            color=MEASURE_COLORS["mutual_info"],
            ms=4,
            ls="",
            label="nonlinear sets",
        )

        # Vertical reference line at Pearson linear AP
        ax.axvline(
            PEARSON_LINEAR_AP,
            color="black",
            linestyle="--",
            lw=0.8,
            alpha=0.6,
        )
        ax.text(
            PEARSON_LINEAR_AP - 0.012,
            (len(prep) - 1) / 2,
            "Pearson |r|, linear sets (0.731)",
            ha="center",
            va="center",
            fontsize=6.5,
            rotation=90,
        )

        # Label the two collapse-on-nonlinear outliers directly (reviewer B-10). Their
        # nonlinear dots sit alone at the far left; text goes to the RIGHT (the gap up
        # to their linear dot is open) so it never collides with the y-axis labels.
        outliers = {"gcmi": "gcmi (copula)", "bspline_mi": "bspline"}
        est_to_row = {est: i for i, est in enumerate(prep["estimator"])}
        for est, text in outliers.items():
            row = est_to_row[est]
            x = float(prep.loc[prep["estimator"] == est, "mean_ap_nonlinear"].iloc[0])
            ax.annotate(
                text,
                (x, row),
                textcoords="offset points",
                xytext=(6, 0),
                ha="left",
                va="center",
                fontsize=5.8,
                color=MEASURE_COLORS["mutual_info"],
            )

        ax.set_yticks(list(y))
        ax.set_yticklabels(prep["estimator"], fontsize=6.5)
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == "mi_reg_k3":
                lbl.set_fontweight("bold")
        ax.set_ylim(-0.7, len(prep) - 0.3)
        ax.set_xlim(0.22, 0.78)
        ax.set_xlabel("mean average precision")
        # make the sort direction explicit
        ax.text(
            0.02,
            0.985,
            "rows sorted by golden AP $\\downarrow$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="0.4",
        )
        ax.invert_yaxis()
        # legend BELOW the axes so it never overlaps the dense top rows of data
        ax.legend(
            frameon=False,
            fontsize=6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=3,
            columnspacing=1.0,
            handletextpad=0.3,
        )

        out = ARTIFACTS / "fig6_mi_wash.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def fig_stopping() -> Path:
    """Fig 7: stopping-rule comparison — recovery F1 vs selected size with CIs."""
    import matplotlib.pyplot as plt

    df = pd.read_parquet(FACTORIAL)
    prep = prep_stopping(df)

    # Extract f1 and k for scatter plot
    f1_data = prep[prep["metric"] == "f1"].set_index("stop_mode")
    k_data = prep[prep["metric"] == "k"].set_index("stop_mode")

    # Identify validation modes
    validation_modes = {"val_fixed_k", "val_threshold"}

    with paper_rc():
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        for mode in sorted(f1_data.index):
            f1_mean = f1_data.loc[mode, "mean"]
            f1_lo = f1_data.loc[mode, "ci_lo"]
            f1_hi = f1_data.loc[mode, "ci_hi"]
            k_mean = k_data.loc[mode, "mean"]
            k_lo = k_data.loc[mode, "ci_lo"]
            k_hi = k_data.loc[mode, "ci_hi"]

            # Choose color based on validation mode
            if mode in validation_modes:
                color = OKABE_ITO["blue"]
            else:
                color = OKABE_ITO["orange"]

            # Draw error bars (whiskers)
            ax.plot(
                [k_lo, k_hi],
                [f1_mean, f1_mean],
                color=color,
                lw=1.0,
            )
            ax.plot(
                [k_mean, k_mean],
                [f1_lo, f1_hi],
                color=color,
                lw=1.0,
            )

            # Draw point
            ax.plot(k_mean, f1_mean, "o", color=color, ms=5)

            # Add label next to point
            ax.annotate(
                mode,
                (k_mean, f1_mean),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=color,
            )

        ax.set_xlabel("mean selected size")
        ax.set_ylabel("mean recovery F1")
        out = ARTIFACTS / "fig7_stopping.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def prep_leaderboard_points(df: pd.DataFrame) -> pd.DataFrame:
    """Per-spec mean recovery F1 tagged with its operator (constant within a spec).

    One row per selector spec (180 golden specs) — the unit the leaderboard strip
    plots. The operator is read off the parquet's ``operator`` column.
    """
    g = golden(df)
    return g.groupby("spec", as_index=False).agg(f1=("f1", "mean"), operator=("operator", "first"))


def fig_leaderboard() -> Path:
    """Fig 8: full 180-spec recovery-F1 distribution by operator, box-and-whisker.

    Replaces the 4-page leaderboard table (reviewer B-14). One box per operator
    over its 60 specs (points overlaid) shows the whole distribution at a glance
    and that the quotient tail is the worst — reprising the operator thesis.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pts = prep_leaderboard_points(pd.read_parquet(FACTORIAL))
    operators = ["difference", "multiplicative", "quotient"]
    rng = np.random.default_rng(0)
    with paper_rc():
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        groups = [
            (op, OPERATOR_COLORS[op], pts.loc[pts["operator"] == op, "f1"].to_numpy())
            for op in operators
        ]
        _boxplot_by_group(ax, groups, rng)
        ax.set_xticks(range(len(operators)))
        ax.set_xticklabels([display_token(op) for op in operators])
        ax.set_xlim(-0.5, len(operators) - 0.5)
        ax.set_ylabel("mean recovery F1 (per spec)")
        ax.set_xlabel("combination operator")
        ax.annotate(
            "quotient tail\n(worst specs)",
            xy=(2, float(pts.loc[pts["operator"] == "quotient", "f1"].min())),
            xytext=(1.25, 0.40),
            fontsize=6.5,
            color=OPERATOR_COLORS["quotient"],
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": OPERATOR_COLORS["quotient"]},
        )
        out = ARTIFACTS / "fig_leaderboard.pdf"
        fig.savefig(out)
        plt.close(fig)
    return out


def _boxplot_by_group(ax, groups: list[tuple[str, str, object]], rng) -> None:
    """Draw one filled box + jittered points per group at integer positions.

    groups: list of (position_label, color, values_array). Shared helper for the
    box-and-whisker figures so their styling stays identical.
    """
    for i, (_, color, vals) in enumerate(groups):
        if len(vals) == 0:
            continue
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
            ms=3,
            alpha=0.5,
            mew=0,
        )


FIGURES: dict[str, Callable[[], Path]] = {
    "fig3_quotient_collapse": fig_quotient_collapse,
    "fig_highdim_real": fig_highdim_real,
    "fig_highdim_downstream": fig_highdim_downstream,
    "fig2_measure_dependence": fig_measure_dependence,
    "fig4_operator_lowdim": fig_operator_lowdim,
    "fig5_operator_aggregation": fig_operator_aggregation,
    "fig6_mi_wash": fig_mi_wash,
    "fig7_stopping": fig_stopping,
    "fig_leaderboard": fig_leaderboard,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.paper_figures", description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ARTIFACTS,
        help=f"output directory for figure PDFs (default: {ARTIFACTS})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    # Module-level-default reassignment: every fig_* builder above reads the
    # module global ARTIFACTS directly (they are dispatched through the
    # FIGURES dict, not called with an explicit path argument), so threading
    # --outdir through each of the 8 signatures would mean changing the
    # Callable[[], Path] contract everywhere. Reassigning the global here,
    # before dispatch, gives every builder the resolved --outdir with a
    # single-line change instead.
    global ARTIFACTS
    args = build_parser().parse_args(argv)
    ARTIFACTS = args.outdir
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for _name, builder in FIGURES.items():
        path = builder()
        print(f"[paper_figures] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
