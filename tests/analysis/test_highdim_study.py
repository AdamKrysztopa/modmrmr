import pandas as pd

from analysis.highdim_study import (
    HIGHDIM_COLUMNS,
    build_parser,
    main,
    make_highdim_synthetic,
    recovery_metrics,
    run_synthetic_sweep,
)
from mechanism.ground_truth import GroundTruth


def test_synthetic_generator_shapes_and_ground_truth_partition():
    X, y, gt = make_highdim_synthetic(p=60, seed=0, n=200)
    assert X.shape == (200, 60) and len(y) == 200
    assert gt.n_features == 60
    assert set(gt.informative) == set(range(5))
    assert set(gt.noise) == set(range(20, 60))


def test_synthetic_generator_is_deterministic():
    X1, _, _ = make_highdim_synthetic(p=60, seed=3, n=200)
    X2, _, _ = make_highdim_synthetic(p=60, seed=3, n=200)
    pd.testing.assert_frame_equal(X1, X2)


def test_recovery_metrics_perfect_and_all_noise():
    gt = GroundTruth(
        informative=(0, 1), codependent=((2,),), noise=(3, 4, 5), dependence="mixed", n_features=6
    )
    perfect = recovery_metrics([0, 1, 2], gt)
    assert perfect["recovery_f1"] == 1.0 and perfect["noise_rate"] == 0.0
    junk = recovery_metrics([3, 4, 5], gt)
    assert junk["recovery_f1"] == 0.0 and junk["noise_rate"] == 1.0


def test_parser_defaults():
    ns = build_parser().parse_args([])
    assert ns.ps == [500, 1000, 2000, 5000, 10000]
    assert ns.seeds == list(range(10))
    assert ns.ks == [5, 10, 20]
    assert ns.out == "results/highdim_study.parquet"


def test_tiny_sweep_end_to_end_with_resume(tmp_path):
    kwargs = dict(ps=[40], seeds=[0], ks=[3], checkpoint_dir=tmp_path)
    df1 = run_synthetic_sweep(**kwargs)
    assert list(df1.columns) == HIGHDIM_COLUMNS
    # 4 operators × 1 k × 1 seed × 1 p
    assert len(df1) == 4
    assert df1["recovery_f1"].between(0, 1).all()
    # rank-based AP/ROC-AUC + recall-over-informative are populated and in range
    assert df1["average_precision"].between(0, 1).all()
    assert df1["roc_auc"].between(0, 1).all()
    assert df1["recall_informative"].between(0, 1).all()
    assert (df1["runtime_s"] > 0).all()
    # second call resumes from the checkpoint and returns identical rows
    df2 = run_synthetic_sweep(**kwargs)
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True), df2.reset_index(drop=True), check_exact=False, atol=1e-12
    )


def test_summary_csv_keeps_multiplicative_and_mult_max_distinct(tmp_path):
    # Regression: the summary groupby must include `aggregation`, else the two
    # operator="multiplicative" specs (mean vs max) collapse into one row and the
    # study's core operator×aggregation comparison is silently averaged away.
    out = tmp_path / "highdim_study.parquet"
    rc = main(["--ps", "40", "--seeds", "0", "--ks", "3", "--real", "none", "--out", str(out)])
    assert rc == 0
    summary = pd.read_csv(tmp_path / "highdim_study_summary.csv")
    # 4 distinct specs for the single (synthetic, p40, k3) cell → 4 summary rows
    assert len(summary) == 4
    mult = summary[summary["operator"] == "multiplicative"]
    assert set(mult["aggregation"]) == {"mean", "max"}
