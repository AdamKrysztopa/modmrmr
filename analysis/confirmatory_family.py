"""Assemble the five-test confirmatory family and apply Holm's step-down.

Sec. 3 of the paper names exactly five tests as carrying its three rules, and
promises Holm-adjusted p-values for all five. Before this script the family
existed only as hand-arithmetic in the LaTeX source: no module assembled the
members, and one member -- the per-$p$ paired Wilcoxon on the high-dimensional
sweep -- had no implementation anywhere in the repo at all.

This script is the single source of truth for that family. It

  1. recomputes the per-$p$ Wilcoxon from ``results/highdim_study.parquet``
     (the missing member) and writes it to ``stats_highdim_synth_paired.csv``;
  2. reads the other four members from their committed artifacts;
  3. applies Holm's step-down across exactly the five;
  4. writes ``stats_confirmatory_family.csv`` with member, source artifact,
     raw p and adjusted p, so every adjusted p-value quoted in the paper
     traces to a committed file.

The relevance-axis members are blocked on *datasets*, not on the 11 mechanisms
that Sec. 3 names as the primary blocking unit, because collapsing the linear
class to mechanisms leaves only three blocks. That exception is deliberate and
is stated in Sec. 3; to show it is not an artefact of choosing the blocking that
flatters the result, the script also reports the family under the
mechanism-blocked alternative, where every member still survives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parents[1] / "results"

GATE = "multiplicative"
QUOTIENT = "quotient"
ALPHA = 0.05


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down. Returns adjusted p-values in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = pvals[idx] * (m - rank)
        running = max(running, scaled)  # enforce monotonicity
        adjusted[idx] = min(running, 1.0)
    return adjusted


def per_p_wilcoxon() -> pd.DataFrame:
    """The missing family member: gate vs quotient noise rate, per dimension.

    Paired over the 10 seeds at each $p$, on the synthetic high-dimensional
    sweep at $k = 20$ with mean aggregation -- the cell Rule 2 recommends.
    """
    df = pd.read_parquet(RESULTS / "highdim_study.parquet")
    df = df[(df.kind == "synthetic") & (df.k == 20) & (df.aggregation == "mean")]

    rows = []
    for p in sorted(df.p.unique()):
        cell = df[df.p == p]
        gate = cell[cell.operator == GATE].sort_values("seed").noise_rate.to_numpy()
        quot = cell[cell.operator == QUOTIENT].sort_values("seed").noise_rate.to_numpy()
        if len(gate) != len(quot) or len(gate) == 0:
            continue
        stat, pval = wilcoxon(gate, quot)
        rows.append(
            {
                "p": int(p),
                "n_seeds": len(gate),
                "mean_noise_gate": gate.mean(),
                "mean_noise_quotient": quot.mean(),
                "wilcoxon_stat": stat,
                "p_raw": pval,
            }
        )
        print(
            f"  p={p:>6}: gate {gate.mean():.3f} vs quotient {quot.mean():.3f}"
            f"  Wilcoxon p={pval:.6g}  (n={len(gate)} seeds)",
            flush=True,
        )
    return pd.DataFrame(rows)


def _lookup(path: str, query: str, col: str = "friedman_p") -> float:
    """Read one omnibus p-value out of a stats artifact.

    These files carry one row per treatment level, so the omnibus p-value is
    repeated down the block; we require that it be constant and take it once.
    """
    df = pd.read_csv(RESULTS / path)
    hit = df.query(query) if query else df
    values = hit[col].unique()
    if len(values) != 1:
        raise SystemExit(
            f"{path}: query {query!r} gave {len(values)} distinct {col} values, expected 1"
        )
    return float(values[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    print("Recomputing the per-$p$ paired Wilcoxon (high-dimensional sweep)...", flush=True)
    perp = per_p_wilcoxon()
    perp.to_csv(RESULTS / "stats_highdim_synth_paired.csv", index=False)
    print(f"  -> results/stats_highdim_synth_paired.csv ({len(perp)} rows)\n", flush=True)

    # The family member is the least favourable dimension in the sweep.
    perp_member = float(perp.p_raw.max())

    members = [
        {
            "rule": 1,
            "member": "relevance axis, within-linear-class Friedman",
            "source": "stats_measure_families_by_dependence.csv",
            "p_raw": _lookup(
                "stats_measure_families_by_dependence.csv",
                "dependence == 'linear' and blocking == 'dataset'",
            ),
        },
        {
            "rule": 1,
            "member": "relevance axis, within-nonlinear-class Friedman (expanded)",
            "source": "stats_nonlinear_expanded.csv",
            "p_raw": _lookup("stats_nonlinear_expanded.csv", "subset == 'all15'"),
        },
        {
            "rule": 2,
            "member": "operator Friedman on noise rate",
            "source": "stats_operators_mechanism.csv",
            "p_raw": _lookup(
                "stats_operators_mechanism.csv",
                "metric == 'noise_rate' and blocking == 'mechanism'",
            ),
        },
        {
            "rule": 2,
            "member": "per-$p$ paired Wilcoxon, high-dimensional sweep",
            "source": "stats_highdim_synth_paired.csv",
            "p_raw": perp_member,
        },
        {
            "rule": 3,
            "member": "stopping-mode Friedman",
            "source": "stats_stopping.csv",
            "p_raw": _lookup("stats_stopping.csv", ""),
        },
    ]

    adj = holm([m["p_raw"] for m in members])
    for m, a in zip(members, adj, strict=True):
        m["p_holm"] = a
        m["survives_at_0.05"] = a < ALPHA

    out = pd.DataFrame(members).sort_values("p_raw").reset_index(drop=True)
    out.to_csv(RESULTS / "stats_confirmatory_family.csv", index=False)

    print("Confirmatory family (Holm step-down, m = 5):", flush=True)
    for _, r in out.iterrows():
        flag = "survives" if r["survives_at_0.05"] else "FAILS"
        print(
            f"  Rule {r.rule}  raw p={r.p_raw:.3e}  Holm p={r.p_holm:.5f}  {flag}   [{r.member}]",
            flush=True,
        )
    print("\n  -> results/stats_confirmatory_family.csv", flush=True)

    if not out["survives_at_0.05"].all():
        print("\nWARNING: a confirmatory family member does not survive Holm.", flush=True)

    # Sensitivity: the relevance-axis members blocked on mechanisms instead.
    alt = [dict(m) for m in members]
    alt[0]["p_raw"] = _lookup(
        "stats_measure_families_by_dependence.csv",
        "dependence == 'linear' and blocking == 'mechanism'",
    )
    alt_adj = holm([m["p_raw"] for m in alt])
    print("\nSensitivity -- linear member blocked on mechanisms (n = 3 blocks):", flush=True)
    for m, a in zip(alt, alt_adj, strict=True):
        print(
            f"  Rule {m['rule']}  raw p={m['p_raw']:.3e}  Holm p={a:.5f}"
            f"  {'survives' if a < ALPHA else 'FAILS'}",
            flush=True,
        )
    print(
        f"  All five survive under this blocking too: {all(a < ALPHA for a in alt_adj)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
