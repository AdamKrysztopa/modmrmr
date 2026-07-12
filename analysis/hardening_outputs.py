"""Derive the paper's significance + cost + baseline artifacts from the hardening runs.

Rerunnable. Consumes the three hardening-run outputs (10-seed factorial, high-dim
p-sweep, external-baseline comparison) and emits tidy CSVs + CD-diagram PDFs that the
figures/prose stages build on. No new library code — composes tested modules
(mechanism.stats, mechanism.cost, mechanism.figures.measure_family).

Reviewer hardening (A-M5 / A-M6 / A-M10):
  * clf/reg TWIN pairs are one mechanism, not two — the operator and measure-family
    omnibus tests are recomputed blocked on ``mechanism`` (11 blocks) alongside the
    dataset-blocked versions (15 blocks) so pseudo-replication is ruled out;
  * stopping rules get a proper Friedman + Nemenyi test and a FAIR single-operating-point
    blind comparator instead of a fixed_k mean smeared over k=1..10;
  * the two CD diagrams plot the SIGNIFICANT tests (operator on noise_rate; measure
    families within the linear dependence class), not the washed-out omnibuses.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from analysis.display_names import display_token
from analysis.paths import ARTIFACTS
from mechanism.cost import runtime_by_family, runtime_vs_p
from mechanism.figures import measure_family
from mechanism.stats import FriedmanNemenyiResult, friedman_nemenyi, plot_critical_difference


def _stats_table(res: FriedmanNemenyiResult) -> pd.DataFrame:
    out = res.avg_ranks.rename("avg_rank").reset_index()
    out = out.rename(columns={out.columns[0]: "treatment"})
    out["friedman_p"] = res.p_value
    out["friedman_stat"] = res.statistic
    out["n_blocks"] = res.n_blocks
    return out.sort_values("avg_rank").reset_index(drop=True)


def _sidebyside_rows(
    res: FriedmanNemenyiResult, *, metric: str, blocking: str, treatment_col: str
) -> list[dict]:
    """One row per treatment for a single (metric, blocking) test — for side-by-side CSVs."""
    return [
        {
            "metric": metric,
            "blocking": blocking,
            treatment_col: str(name),
            "avg_rank": round(float(rank), 4),
            "friedman_p": res.p_value,
            "friedman_stat": res.statistic,
            "n_blocks": res.n_blocks,
        }
        for name, rank in res.avg_ranks.items()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.hardening_outputs", description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory holding the hardening-run inputs and where stats CSVs are written "
        "(default: results)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ARTIFACTS,
        help=f"output directory for the CD-diagram PDFs (default: {ARTIFACTS})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = args.results_dir
    artifacts = args.outdir
    written: list[str] = []

    # ---- Significance over the golden recovery grid -------------------------
    fact = pd.read_parquet(results / "factorial.parquet")
    golden = fact[fact["f1"].notna()].copy()  # golden rows carry recovery; benchmark rows are NaN
    n_golden_datasets = golden["dataset"].nunique()
    # TWIN COLLAPSE (A-M6): clf/reg variants of the same mechanism are one block, not two.
    golden["mechanism"] = golden["dataset"].str.replace(r"_(clf|reg)$", "", regex=True)
    n_mechanisms = golden["mechanism"].nunique()
    golden["family"] = [
        measure_family(r, d) for r, d in zip(golden["relevance"], golden["redundancy"], strict=True)
    ]

    # ---- Operators: F1 and noise_rate, blocked by dataset AND by mechanism --
    # F1 is a WEAK operator signal (the pooled omnibus washes out); noise_rate (fraction of
    # selected features that are pure noise, lower is better) is the load-bearing operator
    # test and is highly significant. We compute all four cells so the paper can show the
    # operator result survives twin-collapse.
    op_side: list[dict] = []
    op_results: dict[tuple[str, str], FriedmanNemenyiResult] = {}
    for metric, higher_is_better in (("f1", True), ("noise_rate", False)):
        for blocking, expected in (("dataset", n_golden_datasets), ("mechanism", n_mechanisms)):
            per = golden.groupby([blocking, "operator"], as_index=False)[metric].mean()
            res = friedman_nemenyi(
                per,
                treatment="operator",
                block=blocking,
                value=metric,
                higher_is_better=higher_is_better,
            )
            # Task-5 carry, now a guard (A-M6): confirm the pivot .dropna() silently dropped
            # no block. n_blocks must equal the count of distinct blocks in this design.
            assert res.n_blocks == expected, (
                f"operator Friedman ({metric}, block={blocking}) dropped blocks: "
                f"n_blocks={res.n_blocks} but {expected} {blocking}s exist"
            )
            op_results[(metric, blocking)] = res
            op_side += _sidebyside_rows(
                res, metric=metric, blocking=blocking, treatment_col="operator"
            )

    # Preserve the historical dataset-blocked F1 artifacts other code may read.
    res_op = op_results[("f1", "dataset")]
    _stats_table(res_op).to_csv(results / "stats_operators.csv", index=False)
    res_op.nemenyi_p.to_csv(results / "stats_operators_nemenyi.csv")
    written += [str(results / "stats_operators.csv"), str(results / "stats_operators_nemenyi.csv")]

    # F2.2 (fix-plan-v3): the prose cites the operator Nemenyi test on the
    # NOISE-RATE metric (quotient-vs-gate p=0.008, quotient-vs-difference
    # p=0.0002; mechanism-blocked, this is the load-bearing operator test), but
    # the historical ``stats_operators_nemenyi.csv`` above holds the F1 version
    # (all p>0.5, non-significant). Ship both metrics under explicit names so
    # every prose citation greps to the right file; the historical filename is
    # kept, unchanged, as the F1 copy for backward compatibility.
    res_op_noise_mech = op_results[("noise_rate", "mechanism")]
    res_op.nemenyi_p.to_csv(results / "stats_operators_nemenyi_f1.csv")
    res_op_noise_mech.nemenyi_p.to_csv(results / "stats_operators_nemenyi_noise.csv")
    written += [
        str(results / "stats_operators_nemenyi_f1.csv"),
        str(results / "stats_operators_nemenyi_noise.csv"),
    ]

    # Side-by-side twin-collapse table (F1 + noise_rate, dataset + mechanism).
    pd.DataFrame(op_side).to_csv(results / "stats_operators_mechanism.csv", index=False)
    written.append(str(results / "stats_operators_mechanism.csv"))

    # CD diagram (A-M5 / 3.3): plot the SIGNIFICANT operator test — noise_rate,
    # mechanism-blocked (twin-collapsed) — NOT the non-significant F1 omnibus.
    # Relabeled to the paper's display names (raw token "multiplicative" -> "gate";
    # fix-plan-v3 F1.1-display) for this rendered PDF only — the CSVs above keep
    # the raw operator tokens.
    display_result = dataclasses.replace(
        res_op_noise_mech,
        avg_ranks=res_op_noise_mech.avg_ranks.rename(index=display_token),
        nemenyi_p=res_op_noise_mech.nemenyi_p.rename(index=display_token, columns=display_token),
    )
    plot_critical_difference(
        display_result,
        artifacts / "cd_operators.pdf",
        title="Operators (noise rate, lower=better; mechanism-blocked)",
    )
    written.append(str(artifacts / "cd_operators.pdf"))

    # ---- Measure families (POOLED) blocked by dataset AND by mechanism ------
    # The measure effect is an INTERACTION with dependence structure (linear-favoring and
    # nonlinear-favoring families point opposite ways), so this pooled main-effect test is
    # expected to wash out under both blockings. Reported only to make the cancellation
    # explicit; the load-bearing test is the per-dependence-class stratification below.
    fam_side: list[dict] = []
    fam_results: dict[str, FriedmanNemenyiResult] = {}
    for blocking in ("dataset", "mechanism"):
        per = golden.groupby([blocking, "family"], as_index=False)["f1"].mean()
        res = friedman_nemenyi(per, treatment="family", block=blocking, value="f1")
        fam_results[blocking] = res
        fam_side += _sidebyside_rows(res, metric="f1", blocking=blocking, treatment_col="family")

    res_fam = fam_results["dataset"]
    _stats_table(res_fam).to_csv(results / "stats_measure_families.csv", index=False)
    res_fam.nemenyi_p.to_csv(results / "stats_measure_families_nemenyi.csv")
    written += [
        str(results / "stats_measure_families.csv"),
        str(results / "stats_measure_families_nemenyi.csv"),
    ]
    pd.DataFrame(fam_side).to_csv(results / "stats_measure_families_mechanism.csv", index=False)
    written.append(str(results / "stats_measure_families_mechanism.csv"))

    # ---- Measure families WITHIN each dependence class ----------------------
    # The correct test for the interaction. Computed dataset-blocked AND mechanism-blocked so
    # the linear within-class result (the certified p=0.002 finding) can be shown twin-robust.
    strat_rows: list[dict] = []
    linear_results: dict[str, FriedmanNemenyiResult] = {}
    for dep in ("linear", "nonlinear", "mixed"):
        sub = golden[golden["dependence"] == dep]
        for blocking in ("dataset", "mechanism"):
            n_blk = sub[blocking].nunique()
            if n_blk < 3:  # Friedman needs >=3 blocks
                strat_rows.append(
                    {"dependence": dep, "blocking": blocking, "n_blocks": n_blk, "friedman_p": None}
                )
                continue
            per = sub.groupby([blocking, "family"], as_index=False)["f1"].mean()
            res = friedman_nemenyi(per, treatment="family", block=blocking, value="f1")
            if dep == "linear":
                linear_results[blocking] = res
            best = res.avg_ranks.idxmin()
            for fam, rank in res.avg_ranks.items():
                strat_rows.append(
                    {
                        "dependence": dep,
                        "blocking": blocking,
                        "n_blocks": res.n_blocks,
                        "family": fam,
                        "avg_rank": round(float(rank), 3),
                        "best_family": best,
                        "friedman_p": res.p_value,
                    }
                )
    pd.DataFrame(strat_rows).to_csv(
        results / "stats_measure_families_by_dependence.csv", index=False
    )
    written.append(str(results / "stats_measure_families_by_dependence.csv"))

    # CD diagram (A-M5 / 3.3): plot the SIGNIFICANT within-class measure test — measure
    # families on the LINEAR dependence subset — NOT the pooled non-significant omnibus.
    # The certified finding is dataset-blocked (6 blocks, p=0.002); mechanism-blocking
    # collapses the linear subset to only 3 blocks (still significant, p=0.05, but
    # underpowered), so we plot the certified dataset-blocked test.
    res_lin = linear_results.get("dataset") or linear_results.get("mechanism")
    plot_critical_difference(
        res_lin,
        artifacts / "cd_measures.pdf",
        title="Measure families within LINEAR dependence (recovery F1)",
    )
    written.append(str(artifacts / "cd_measures.pdf"))

    # ---- Stopping rules (A-M10) ---------------------------------------------
    # (a) Proper inference: Friedman over the 4 stop modes, blocked by mechanism (twins
    #     collapsed), value=F1 (higher is better). Emits stats + Nemenyi + CD diagram.
    per_stop = golden.groupby(["mechanism", "stop_mode"], as_index=False)["f1"].mean()
    res_stop = friedman_nemenyi(
        per_stop, treatment="stop_mode", block="mechanism", value="f1", higher_is_better=True
    )
    _stats_table(res_stop).to_csv(results / "stats_stopping.csv", index=False)
    res_stop.nemenyi_p.to_csv(results / "stats_stopping_nemenyi.csv")
    plot_critical_difference(
        res_stop,
        artifacts / "cd_stopping.pdf",
        title="Stopping rules (recovery F1; mechanism-blocked)",
    )
    written += [
        str(results / "stats_stopping.csv"),
        str(results / "stats_stopping_nemenyi.csv"),
        str(artifacts / "cd_stopping.pdf"),
    ]

    # (b) FAIR blind comparator: a single blind fixed-k operating point (k=5 and k=10,
    #     separately) versus the validation-selected modes — NOT fixed_k smeared over
    #     k=1..10. "selected size" is the realized number of features kept (k), which for
    #     the threshold modes is the count the rule actually stopped at. Reported as the
    #     grand mean of per-mechanism means (equal weight per mechanism, twin-collapsed).
    def _operating_point(rows: pd.DataFrame, label: str) -> dict:
        per_mech = rows.groupby("mechanism").agg(
            mean_f1=("f1", "mean"), mean_selected_size=("k", "mean")
        )
        return {
            "mode": label,
            "mean_f1": round(float(per_mech["mean_f1"].mean()), 4),
            "mean_selected_size": round(float(per_mech["mean_selected_size"].mean()), 3),
            "n_mechanisms": int(per_mech.shape[0]),
        }

    fixed = golden[golden["stop_mode"] == "fixed_k"]
    comparator_rows = [
        _operating_point(fixed[fixed["k"] == 5], "fixed_k@5"),
        _operating_point(fixed[fixed["k"] == 10], "fixed_k@10"),
        _operating_point(golden[golden["stop_mode"] == "val_fixed_k"], "val_fixed_k"),
        _operating_point(golden[golden["stop_mode"] == "val_threshold"], "val_threshold"),
        _operating_point(golden[golden["stop_mode"] == "threshold"], "threshold"),
    ]
    pd.DataFrame(comparator_rows).to_csv(results / "stopping_fair_comparator.csv", index=False)
    written.append(str(results / "stopping_fair_comparator.csv"))

    # ---- Cost ---------------------------------------------------------------
    runtime_by_family(fact).to_csv(results / "cost_summary.csv", index=False)
    high = pd.read_parquet(results / "highdim_study.parquet")
    runtime_vs_p(high[high["kind"] == "synthetic"]).to_csv(results / "cost_vs_p.csv", index=False)
    written += [str(results / "cost_summary.csv"), str(results / "cost_vs_p.csv")]

    # ---- External-baseline summary (Task-7 carry: report per-method coverage)
    base = pd.read_csv(results / "baseline_comparison.csv")
    base_summary = (
        base.groupby(["method", "source"], as_index=False)
        .agg(
            mean_recovery_f1=("recovery_f1", "mean"),
            mean_noise_rate=("noise_rate", "mean"),
            n_datasets=("dataset", "nunique"),
            n_rows=("recovery_f1", "size"),
        )
        .sort_values("mean_recovery_f1", ascending=False)
        .reset_index(drop=True)
    )
    base_summary.to_csv(results / "baseline_comparison_summary.csv", index=False)
    written.append(str(results / "baseline_comparison_summary.csv"))

    for path in written:
        print(f"WROTE {path}", flush=True)

    # ---- Console summary of every reported p-value --------------------------
    print("\n=== operator Friedman (treatment=operator) ===", flush=True)
    for metric in ("noise_rate", "f1"):
        ds = op_results[(metric, "dataset")]
        me = op_results[(metric, "mechanism")]
        print(
            f"  {metric:11s}: dataset  p={ds.p_value:.2e} (stat={ds.statistic:.2f}, "
            f"n={ds.n_blocks}) | mechanism p={me.p_value:.2e} (stat={me.statistic:.2f}, "
            f"n={me.n_blocks})",
            flush=True,
        )
    print("=== measure families, LINEAR within-class ===", flush=True)
    for blocking, res in linear_results.items():
        print(
            f"  {blocking:9s}: p={res.p_value:.2e} (stat={res.statistic:.2f}, n={res.n_blocks}) "
            f"ranks={dict(res.avg_ranks.round(3))}",
            flush=True,
        )
    print("=== measure families, POOLED (expected wash-out) ===", flush=True)
    for blocking, res in fam_results.items():
        print(f"  {blocking:9s}: p={res.p_value:.2e} (n={res.n_blocks})", flush=True)
    print("=== stopping rules (Friedman) ===", flush=True)
    print(
        f"  p={res_stop.p_value:.2e} (stat={res_stop.statistic:.2f}, n={res_stop.n_blocks}) "
        f"ranks={dict(res_stop.avg_ranks.round(3))}",
        flush=True,
    )
    print("  Nemenyi pairwise p-values:", flush=True)
    print(res_stop.nemenyi_p.round(4).to_string(), flush=True)
    print("=== fair blind comparator ===", flush=True)
    print(pd.DataFrame(comparator_rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
