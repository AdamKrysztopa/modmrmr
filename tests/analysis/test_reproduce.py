from pathlib import Path

import pandas as pd
import pytest

from analysis.reproduce import main, regenerate
from analysis.schema import RESULT_COLUMNS, make_synthetic_results


def test_regenerate_writes_all_artifacts(tmp_path: Path) -> None:
    written = regenerate(make_synthetic_results(), tmp_path)
    names = {p.name for p in written}
    # both tasks x four figures
    for task in ("classification", "regression"):
        assert f"cd_{task}.png" in names
        assert f"grid_{task}.png" in names
        assert f"auc_k_{task}.png" in names
        assert f"stability_{task}.png" in names
        assert f"win_rank_{task}.tex" in names
    assert "design_space.tex" in names
    assert "decision_guide.tex" in names
    assert "decision_guide.json" in names
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_main_defaults_to_synthetic(tmp_path: Path) -> None:
    rc = main(["--outdir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "design_space.tex").exists()


def test_regenerate_rejects_empty_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        regenerate(pd.DataFrame(columns=RESULT_COLUMNS), tmp_path)


def test_regenerate_handles_single_task_results(tmp_path: Path) -> None:
    # A classification-only results file must render cleanly (not KeyError) and
    # skip the absent regression task.
    results = make_synthetic_results()
    clf_only = results[results["task"] == "classification"]
    written = regenerate(clf_only, tmp_path)
    names = {p.name for p in written}
    assert "cd_classification.png" in names
    assert "win_rank_classification.tex" in names
    assert "cd_regression.png" not in names
    assert "win_rank_regression.tex" not in names
