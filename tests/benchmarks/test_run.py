import pandas as pd
import pytest
from sklearn.datasets import make_classification

from benchmarks.run import build_parser, main


@pytest.fixture()
def _no_download(monkeypatch):
    def fake_load(name):
        Xa, ya = make_classification(n_samples=120, n_features=12, n_informative=5, random_state=0)
        return (
            pd.DataFrame(Xa, columns=[f"f{i}" for i in range(12)]),
            pd.Series(ya),
            "classification",
        )

    # run_grid resolves load_dataset from benchmarks.protocol, so patch it there.
    monkeypatch.setattr("benchmarks.protocol.load_dataset", fake_load)


def test_parser_has_expected_options():
    parser = build_parser()
    ns = parser.parse_args(
        [
            "--datasets",
            "synthetic_clf",
            "--cells",
            "skb_f",
            "--ks",
            "2",
            "5",
            "--cv",
            "3",
            "--seeds",
            "0",
            "--out",
            "x.csv",
        ]
    )
    assert ns.datasets == ["synthetic_clf"]
    assert ns.cells == ["skb_f"]
    assert ns.ks == [2, 5]
    assert ns.cv == 3
    assert ns.seeds == [0]
    assert ns.out == "x.csv"


def test_list_flag_works_without_required_run_args(capsys):
    # --list must short-circuit before the datasets/cells/out requirement, so the
    # documented `python -m benchmarks.run --list` invocation works standalone.
    rc = main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "datasets:" in out
    assert "cells:" in out


def test_run_without_required_args_errors(capsys):
    with pytest.raises(SystemExit):
        main(["--cells", "skb_f"])  # missing --datasets/--out


def test_main_writes_csv_results(tmp_path, _no_download):
    out = tmp_path / "results.csv"
    rc = main(
        [
            "--datasets",
            "synthetic_clf",
            "--cells",
            "skb_f",
            "--ks",
            "2",
            "5",
            "--cv",
            "3",
            "--seeds",
            "0",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    df = pd.read_csv(out)
    assert {"dataset", "method", "task", "seed", "k", "learner", "metric", "score"} <= set(
        df.columns
    )
    assert len(df) > 0


def test_main_writes_parquet_results(tmp_path, _no_download):
    out = tmp_path / "results.parquet"
    rc = main(
        [
            "--datasets",
            "synthetic_clf",
            "--cells",
            "skb_f",
            "--ks",
            "5",
            "--cv",
            "3",
            "--seeds",
            "0",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 0
