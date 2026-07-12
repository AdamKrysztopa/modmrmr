from pathlib import Path

import pandas as pd

from analysis.figures import (
    auc_k_barchart,
    cd_diagram,
    grid_heatmap,
    stability_accuracy_scatter,
)


def test_cd_diagram_writes_file(synthetic_results: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "cd_classification.png"
    result = cd_diagram(synthetic_results, "classification", out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_grid_heatmap_writes_file(synthetic_results: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "grid_regression.png"
    result = grid_heatmap(synthetic_results, "regression", out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_auc_k_barchart_writes_file(synthetic_results: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "auc_k_classification.png"
    assert auc_k_barchart(synthetic_results, "classification", out) == out
    assert out.exists() and out.stat().st_size > 0


def test_stability_scatter_writes_file(synthetic_results: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "stability_regression.png"
    assert stability_accuracy_scatter(synthetic_results, "regression", out) == out
    assert out.exists() and out.stat().st_size > 0
