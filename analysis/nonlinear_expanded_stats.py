"""Within-class Rule-1 test on the expanded 15-generator nonlinear suite.

Combines the 11 new nonlinear generators (``results/factorial_nonlin.parquet``)
with the 4 original nonlinear golden datasets from the main factorial
(``results/factorial.parquet``; dependence == "nonlinear": parabola, radial,
sine, xor) and runs the within-class Friedman test over measure families
(linear / mixed / nonlinear via ``mechanism.figures.measure_family``), blocked
by dataset on per-dataset mean recovery F1, with the Nemenyi post-hoc contrast
of the nonlinear vs linear families. Computed twice: over all 15 generators
and with the ``xor`` floor case removed (14 blocks), showing the result is not
an ``xor`` artifact. Writes ``results/stats_nonlinear_expanded.csv``.

Also logs the raw-row family mean F1 (nonlinear 0.539 vs linear 0.261 — the
paper's "2.1x" gap) so every cited number for this study appears in the run log.

Emits incremental progress even though this is a fast, non-checkpointed step: a
long silent run is indistinguishable from a hang.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mechanism.figures import measure_family
from mechanism.stats import friedman_nemenyi

ORIGINAL_NONLINEAR_DEPENDENCE = "nonlinear"
FLOOR_CASE = "xor"


def log(msg: str) -> None:
    print(msg, flush=True)


def _load_expanded_suite(nonlin_parquet: Path, factorial_parquet: Path) -> pd.DataFrame:
    """Concatenate the 11 new generators with the 4 original nonlinear golden sets."""
    log(f"reading {nonlin_parquet}")
    nonlin = pd.read_parquet(nonlin_parquet)
    nonlin = nonlin[nonlin["f1"].notna()]
    log(f"  new generators: {sorted(nonlin['dataset'].unique())} ({len(nonlin)} rows)")

    log(f"reading {factorial_parquet}")
    fact = pd.read_parquet(factorial_parquet)
    golden = fact[fact["f1"].notna()]  # golden rows carry recovery; benchmark rows are NaN
    original = golden[golden["dependence"] == ORIGINAL_NONLINEAR_DEPENDENCE]
    log(f"  original nonlinear: {sorted(original['dataset'].unique())} ({len(original)} rows)")

    suite = pd.concat([nonlin, original], ignore_index=True)
    suite["family"] = [
        measure_family(rel, red)
        for rel, red in zip(suite["relevance"], suite["redundancy"], strict=True)
    ]
    n_datasets = suite["dataset"].nunique()
    log(f"expanded suite: {n_datasets} generators, {len(suite)} rows")
    return suite


def run(nonlin_parquet: Path, factorial_parquet: Path, out_path: Path) -> pd.DataFrame:
    suite = _load_expanded_suite(nonlin_parquet, factorial_parquet)
    n_datasets = suite["dataset"].nunique()

    # Paper's "2.1x" family gap: raw-row mean F1 per measure family over the suite.
    raw_means = suite.groupby("family")["f1"].mean()
    log("raw-row family mean F1 (the 2.1x claim):")
    for fam, mean in raw_means.sort_values(ascending=False).items():
        log(f"  {fam:9s}: {mean:.3f}")

    rows: list[dict] = []
    subsets = [
        ("all15", suite),
        (f"no_{FLOOR_CASE}", suite[suite["dataset"] != FLOOR_CASE]),
    ]
    for i, (label, sub) in enumerate(subsets, start=1):
        expected = sub["dataset"].nunique()
        per = sub.groupby(["dataset", "family"], as_index=False)["f1"].mean()
        res = friedman_nemenyi(
            per, treatment="family", block="dataset", value="f1", higher_is_better=True
        )
        # Guard (mirrors analysis.hardening_outputs): the pivot .dropna() must not
        # silently drop a block — every generator has all three families.
        assert res.n_blocks == expected, (
            f"expanded-nonlinear Friedman ({label}) dropped blocks: "
            f"n_blocks={res.n_blocks} but {expected} datasets exist"
        )
        rows.append(
            {
                "subset": label,
                "n_blocks": res.n_blocks,
                "friedman_p": res.p_value,
                "nemenyi_nonlinear_vs_linear": float(res.nemenyi_p.loc["nonlinear", "linear"]),
                "rank_nonlinear": float(res.avg_ranks["nonlinear"]),
                "rank_linear": float(res.avg_ranks["linear"]),
            }
        )
        log(
            f"  [{i}/{len(subsets)}] {label:7s}: n_blocks={res.n_blocks} "
            f"friedman_p={res.p_value:.2e} "
            f"nemenyi(nonlinear vs linear)={res.nemenyi_p.loc['nonlinear', 'linear']:.2e} "
            f"ranks={dict(res.avg_ranks.round(3))}"
        )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "subset",
            "n_blocks",
            "friedman_p",
            "nemenyi_nonlinear_vs_linear",
            "rank_nonlinear",
            "rank_linear",
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log(f"wrote {len(out_df)} rows -> {out_path}")
    log(
        f"note: the {FLOOR_CASE} floor case is removed in the second subset because its signal "
        f"is purely interactional — every univariate measure fails on it — so the no_{FLOOR_CASE} "
        f"row shows the result is not a {FLOOR_CASE} artifact. n_datasets={n_datasets}."
    )
    return out_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.nonlinear_expanded_stats", description=__doc__)
    parser.add_argument("--nonlin-parquet", type=str, default="results/factorial_nonlin.parquet")
    parser.add_argument("--factorial-parquet", type=str, default="results/factorial.parquet")
    parser.add_argument("--out", type=str, default="results/stats_nonlinear_expanded.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(Path(args.nonlin_parquet), Path(args.factorial_parquet), Path(args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
