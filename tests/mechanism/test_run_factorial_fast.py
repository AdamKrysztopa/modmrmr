"""Tests for the ``mechanism.run_factorial_fast`` CLI.

Only TINY configs are exercised here (the full study is a controller-run later
phase). Beyond the smoke coverage that mirrors ``test_run_factorial``, these
tests pin the two behaviors unique to the fast CLI: per-shard checkpoint
parquets and resume-from-checkpoint (with ``--fresh`` forcing recomputation).
"""

from __future__ import annotations

import pandas as pd
import pytest

from mechanism.datasets import list_mechanism_datasets
from mechanism.factorial import CANONICAL_NAMED, FULL_FACTORIAL
from mechanism.factorial_protocol import FACTORIAL_COLUMNS
from mechanism.run_factorial_fast import build_parser, main


def _read_results(out_path):
    if out_path.exists():
        return pd.read_parquet(out_path) if out_path.suffix == ".parquet" else pd.read_csv(out_path)
    csv_fallback = out_path.with_suffix(".csv")
    assert csv_fallback.exists(), f"neither {out_path} nor its csv fallback exists"
    return pd.read_csv(csv_fallback)


def _read_shard(path):
    if path.exists():
        return pd.read_parquet(path)
    csv_fallback = path.with_suffix(".csv")
    assert csv_fallback.exists(), f"neither {path} nor its csv fallback exists"
    return pd.read_csv(csv_fallback)


def _shard_exists(shards_dir, dataset, seed) -> bool:
    p = shards_dir / f"{dataset}__{seed}.parquet"
    return p.exists() or p.with_suffix(".csv").exists()


_TINY_ARGS = [
    "--datasets",
    "quotient_trap_reg",
    "--benchmark-datasets",
    "none",
    "--specs",
    "f|pearson_abs|multiplicative|max",
    "f|pearson_abs|quotient|mean",
    "--ks",
    "1",
    "2",
    "--thresholds",
    "0.0",
    "--seeds",
    "0",
    "--jobs",
    "1",
]


def test_build_parser_defaults():
    parser = build_parser()
    ns = parser.parse_args([])
    assert ns.datasets is None
    assert ns.benchmark_datasets == ["breast_cancer", "synthetic_clf", "diabetes", "friedman1"]
    assert ns.specs is None
    assert ns.ks == [1, 2, 3, 5, 8, 10]
    assert ns.thresholds == [0.0, 0.05, 0.1, 0.2]
    assert ns.seeds == [0, 1, 2]
    assert ns.out == "results/factorial.parquet"
    assert ns.figures_dir == "results/figures"
    assert ns.jobs == -1
    assert ns.fresh is False
    assert ns.list is False


def test_list_returns_zero_without_required_args(capsys):
    rc = main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    for spec in FULL_FACTORIAL:
        assert spec.label in out
    for name in CANONICAL_NAMED:
        assert name in out
    for name in list_mechanism_datasets():
        assert name in out


def test_tiny_end_to_end_writes_shards_and_final_outputs(tmp_path):
    out = tmp_path / "f.parquet"
    figures_dir = tmp_path / "figs"
    rc = main([*_TINY_ARGS, "--out", str(out), "--figures-dir", str(figures_dir)])
    assert rc == 0

    df = _read_results(out)
    assert list(df.columns) == FACTORIAL_COLUMNS
    assert len(df) > 0
    assert set(df["dataset"].unique()) == {"quotient_trap_reg"}

    # per-shard checkpoint was written under results/_shards/
    shards_dir = out.parent / "_shards"
    assert shards_dir.exists()
    assert _shard_exists(shards_dir, "quotient_trap_reg", 0)

    # final tables
    assert out.with_name(f"{out.stem}_summary.csv").exists()
    assert out.with_name(f"{out.stem}_decision_guide.csv").exists()
    figure_files = list(figures_dir.glob("*.png"))
    assert len(figure_files) > 0


def test_second_invocation_resumes_from_checkpoints(tmp_path, capsys):
    out = tmp_path / "f.parquet"
    figures_dir = tmp_path / "figs"

    rc = main([*_TINY_ARGS, "--out", str(out), "--figures-dir", str(figures_dir)])
    assert rc == 0
    first = _read_results(out)
    capsys.readouterr()  # drain

    # Second run: checkpoint already exists -> shard is RESUMED, not recomputed.
    rc = main([*_TINY_ARGS, "--out", str(out), "--figures-dir", str(figures_dir)])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "RESUMED from checkpoint" in out_text

    second = _read_results(out)
    pd.testing.assert_frame_equal(
        first.drop(columns=["runtime_s"]), second.drop(columns=["runtime_s"])
    )


def test_fresh_flag_recomputes_and_does_not_resume(tmp_path, capsys):
    out = tmp_path / "f.parquet"
    figures_dir = tmp_path / "figs"

    rc = main([*_TINY_ARGS, "--out", str(out), "--figures-dir", str(figures_dir)])
    assert rc == 0
    capsys.readouterr()  # drain

    # --fresh: existing checkpoint is ignored, so no RESUMED line is printed.
    rc = main([*_TINY_ARGS, "--out", str(out), "--figures-dir", str(figures_dir), "--fresh"])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "RESUMED from checkpoint" not in out_text
    assert "--fresh" in out_text


def test_resume_reuses_a_planted_checkpoint_verbatim(tmp_path, capsys):
    """A checkpoint on disk is trusted verbatim: a planted sentinel-row shard is
    returned as-is (proving the shard computation was genuinely skipped, not merely
    recomputed to the same value)."""
    out = tmp_path / "f.parquet"
    figures_dir = tmp_path / "figs"
    shards_dir = out.parent / "_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # First real run to get a valid shard schema, then mutate + re-plant it.
    rc = main([*_TINY_ARGS, "--out", str(out), "--figures-dir", str(figures_dir)])
    assert rc == 0
    shard_path = shards_dir / "quotient_trap_reg__0.parquet"
    planted = _read_shard(shard_path)
    sentinel = planted.iloc[[0]].copy()
    sentinel["downstream_score"] = -123.0
    try:
        sentinel.to_parquet(shard_path, index=False)
    except Exception:  # noqa: BLE001 - parquet engine missing; plant csv sidecar instead
        sentinel.to_csv(shard_path.with_suffix(".csv"), index=False)
    capsys.readouterr()

    rc = main([*_TINY_ARGS, "--out", str(out), "--figures-dir", str(figures_dir)])
    assert rc == 0
    df = _read_results(out)
    assert (df["downstream_score"] == -123.0).any(), (
        "planted sentinel row was not returned verbatim; the shard was recomputed "
        "instead of resumed from its checkpoint"
    )


def test_unknown_dataset_or_spec_errors():
    with pytest.raises(SystemExit):
        main(
            [
                "--datasets",
                "not_a_real_dataset",
                "--benchmark-datasets",
                "none",
                "--specs",
                "ModMRMR",
                "--ks",
                "1",
                "--seeds",
                "0",
                "--out",
                "x.csv",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--datasets",
                "quotient_trap_reg",
                "--benchmark-datasets",
                "none",
                "--specs",
                "not_a_real_spec",
                "--ks",
                "1",
                "--seeds",
                "0",
                "--out",
                "x.csv",
            ]
        )
