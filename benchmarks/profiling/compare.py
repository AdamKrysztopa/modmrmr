"""Before/after comparison and Rust decision-gate evaluation.

Usage::

    uv run python -m benchmarks.profiling.compare
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path("results/profiling")

# Amdahl threshold: a kernel below this share of end-to-end time cannot justify
# a toolchain dependency, because even infinite speedup barely moves the total.
_DOMINANCE_THRESHOLD = 0.50
# Below this fitted exponent, cost grows slowly enough that a constant-factor
# rewrite is not the right lever.
_EXPONENT_THRESHOLD = 1.5
# Scorers whose cost is a third-party model fit. Rust cannot touch these:
# tree_r2 fits a 200-tree sklearn forest per pair; relieff fits skrebate.
_THIRD_PARTY_BOUND = frozenset({"tree_r2", "relieff"})


def fit_exponent(df: pd.DataFrame) -> pd.DataFrame:
    """Fit ``log(t) = a + b*log(n)`` per (scorer, data_kind).

    Args:
        df: A ``scorer_scaling`` frame.

    Returns:
        Columns ``scorer``, ``data_kind``, ``exponent``, ``n_points``. Groups
        with fewer than two measured points get ``exponent = NaN``.
    """
    measured = df[(df["status"] == "ok") & df["median_s"].notna()]
    records = []
    for (scorer, kind), group in measured.groupby(["scorer", "data_kind"], sort=True):
        if len(group) < 2:
            records.append(
                {"scorer": scorer, "data_kind": kind, "exponent": np.nan, "n_points": len(group)}
            )
            continue
        slope = np.polyfit(
            np.log(group["n"].to_numpy(dtype=float)),
            np.log(np.maximum(group["median_s"].to_numpy(dtype=float), 1e-12)),
            1,
        )[0]
        records.append(
            {
                "scorer": scorer,
                "data_kind": kind,
                "exponent": float(slope),
                "n_points": len(group),
            }
        )
    return pd.DataFrame(records)


def speedup_table(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Join baseline and post-optimization timings and report the ratio."""
    keys = [c for c in ("scorer", "data_kind", "n", "p") if c in before.columns]
    merged = before.merge(after, on=keys, suffixes=("_before", "_after"))
    merged = merged[(merged["status_before"] == "ok") & (merged["status_after"] == "ok")].copy()
    merged["speedup"] = merged["median_s_before"] / merged["median_s_after"]
    return merged[[*keys, "median_s_before", "median_s_after", "speedup"]].reset_index(drop=True)


def evaluate_gate(after_end_to_end: pd.DataFrame, exponents: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the three Rust-gate conditions per scorer.

    Args:
        after_end_to_end: Post-optimization ``end_to_end`` frame.
        exponents: Output of :func:`fit_exponent`.

    Returns:
        One row per scorer with the three conditions and the combined verdict.
    """
    measured = after_end_to_end[after_end_to_end["status"] == "ok"]
    totals = measured.groupby("scorer")["median_s"].sum()
    share = totals / totals.sum() if totals.sum() > 0 else totals * 0.0

    worst_exponent = exponents.groupby("scorer")["exponent"].max()

    records = []
    for scorer in totals.index:
        dominates = bool(share.get(scorer, 0.0) >= _DOMINANCE_THRESHOLD)
        exponent = float(worst_exponent.get(scorer, np.nan))
        superlinear = bool(exponent >= _EXPONENT_THRESHOLD)
        own_code = scorer not in _THIRD_PARTY_BOUND
        records.append(
            {
                "scorer": scorer,
                "time_share": float(share.get(scorer, 0.0)),
                "dominates": dominates,
                "exponent": exponent,
                "superlinear": superlinear,
                "own_code": own_code,
                "passes_gate": dominates and superlinear and own_code,
            }
        )
    return pd.DataFrame(records).sort_values("time_share", ascending=False).reset_index(drop=True)


def main() -> None:
    """Print the speedup tables and the gate verdict."""
    for name in ("scorer_scaling", "driver_scaling", "end_to_end"):
        before_path = DEFAULT_DIR / f"{name}.csv"
        after_path = DEFAULT_DIR / f"{name}_after.csv"
        if not after_path.exists():
            print(f"[{name}] no _after file yet — run the sweep first", flush=True)
            continue
        before = pd.read_csv(before_path)
        after = pd.read_csv(after_path)
        print(f"\n=== {name}: speedup ===", flush=True)
        print(speedup_table(before, after).to_string(index=False), flush=True)

    after_scaling = DEFAULT_DIR / "scorer_scaling_after.csv"
    after_e2e = DEFAULT_DIR / "end_to_end_after.csv"
    if after_scaling.exists() and after_e2e.exists():
        exponents = fit_exponent(pd.read_csv(after_scaling))
        print("\n=== fitted scaling exponents ===", flush=True)
        print(exponents.to_string(index=False), flush=True)
        verdicts = evaluate_gate(pd.read_csv(after_e2e), exponents)
        print("\n=== Rust decision gate ===", flush=True)
        print(verdicts.to_string(index=False), flush=True)
        if verdicts["passes_gate"].any():
            winners = verdicts.loc[verdicts["passes_gate"], "scorer"].tolist()
            print(f"\nGATE PASSES for: {', '.join(winners)}", flush=True)
        else:
            print(
                "\nGATE FAILS — no kernel is simultaneously dominant, "
                "superlinear, and not third-party-bound. Rust is not justified.",
                flush=True,
            )


if __name__ == "__main__":
    main()
