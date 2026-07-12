"""Tests for paper figure prep functions — assert the verified headline numbers."""

from pathlib import Path

import pandas as pd
import pytest

HIGHDIM = Path("results/highdim_study.parquet")

pytestmark = pytest.mark.skipif(not HIGHDIM.exists(), reason="results parquet not present")


@pytest.fixture(scope="module")
def highdim() -> pd.DataFrame:
    return pd.read_parquet(HIGHDIM)


def _val(prep: pd.DataFrame, variant: str, p: int, metric: str) -> float:
    row = prep[(prep["variant"] == variant) & (prep["p"] == p) & (prep["metric"] == metric)]
    assert len(row) == 1
    return float(row["mean"].iloc[0])


class TestPrepQuotientCollapse:
    def test_headline_noise_rates(self, highdim):
        from analysis.paper_figures import prep_quotient_collapse

        # 10-seed values (fix-plan 5.1). Quotient climbs; the bounded gate is flat.
        prep = prep_quotient_collapse(highdim)
        assert _val(prep, "quotient", 500, "noise_rate") == pytest.approx(0.69, abs=0.02)
        assert _val(prep, "quotient", 10000, "noise_rate") == pytest.approx(0.78, abs=0.02)
        assert _val(prep, "multiplicative", 500, "noise_rate") == pytest.approx(0.21, abs=0.02)
        assert _val(prep, "multiplicative", 10000, "noise_rate") == pytest.approx(0.26, abs=0.02)

    def test_reg_quotient_arm_removes_most_collapse(self, highdim):
        from analysis.paper_figures import prep_quotient_collapse

        # fix-plan 5.2 / A-M8: regularizing the denominator (D/(W+eps)) recovers
        # recall and cuts noise far below the raw quotient, but not below the gate.
        prep = prep_quotient_collapse(highdim)
        regq_noise = _val(prep, "reg_quotient", 10000, "noise_rate")
        quot_noise = _val(prep, "quotient", 10000, "noise_rate")
        gate_noise = _val(prep, "multiplicative", 10000, "noise_rate")
        assert regq_noise == pytest.approx(0.40, abs=0.03)
        assert regq_noise < quot_noise - 0.2  # much cleaner than the raw quotient
        assert gate_noise < regq_noise - 0.05  # but the gate is cleaner still

    def test_quotient_noise_rises_with_p_and_mult_is_flat(self, highdim):
        from analysis.paper_figures import prep_quotient_collapse

        prep = prep_quotient_collapse(highdim)
        quot = prep[(prep["variant"] == "quotient") & (prep["metric"] == "noise_rate")]
        quot = quot.sort_values("p")["mean"].tolist()
        assert quot[-1] > quot[0]  # rising
        mult = prep[(prep["variant"] == "multiplicative") & (prep["metric"] == "noise_rate")]
        assert mult["mean"].max() <= 0.28  # dimension-invariant

    def test_recall_coverage(self, highdim):
        from analysis.paper_figures import prep_quotient_collapse

        prep = prep_quotient_collapse(highdim)
        rec = prep[prep["metric"] == "recall_informative"].groupby("variant")["mean"].mean()
        assert rec["quotient"] == pytest.approx(0.32, abs=0.03)
        assert rec["multiplicative"] == pytest.approx(0.68, abs=0.03)
        assert rec["difference"] == pytest.approx(0.66, abs=0.03)
        assert rec["reg_quotient"] == pytest.approx(0.68, abs=0.03)
        assert rec["mult_max"] == pytest.approx(0.56, abs=0.03)

    def test_ci_bounds_bracket_mean(self, highdim):
        from analysis.paper_figures import prep_quotient_collapse

        prep = prep_quotient_collapse(highdim)
        assert (prep["ci_lo"] <= prep["mean"]).all()
        assert (prep["mean"] <= prep["ci_hi"]).all()


class TestPrepHighdimReal:
    def test_madelon_gap(self, highdim):
        from analysis.paper_figures import prep_highdim_real

        prep = prep_highdim_real(highdim)

        def val(ds, variant):
            r = prep[(prep["dataset"] == ds) & (prep["variant"] == variant)]
            return float(r["mean"].iloc[0])

        # 10-seed real-arm values (fix-plan F2.3 / H-5, seeds 0-9). The 3-seed
        # values these superseded were 0.838/0.597 (madelon) and 0.767/0.703
        # (arcene); the gate's decisive win on both holds up at 10 seeds.
        assert val("madelon", "multiplicative") == pytest.approx(0.841, abs=0.01)
        assert val("madelon", "quotient") == pytest.approx(0.590, abs=0.01)
        assert val("arcene", "multiplicative") == pytest.approx(0.767, abs=0.01)
        assert val("arcene", "quotient") == pytest.approx(0.692, abs=0.01)


HIGHDIM_DOWNSTREAM_SYNTH = Path("results/highdim_downstream_synth.csv")

downstream_synth_missing = pytest.mark.skipif(
    not HIGHDIM_DOWNSTREAM_SYNTH.exists(), reason="downstream-synth csv absent"
)


@downstream_synth_missing
def test_prep_highdim_downstream_synth_ci_brackets_mean():
    from analysis.paper_figures import prep_highdim_downstream_synth

    df = pd.read_csv(HIGHDIM_DOWNSTREAM_SYNTH)
    prep = prep_highdim_downstream_synth(df)
    assert {"difference", "multiplicative", "quotient", "reg_quotient"} <= set(prep["operator"])
    assert (prep["ci_lo"] <= prep["mean"]).all()
    assert (prep["mean"] <= prep["ci_hi"]).all()


@downstream_synth_missing
def test_fig_highdim_downstream_writes_pdf(tmp_path, monkeypatch):
    import analysis.paper_figures as pf

    monkeypatch.setattr(pf, "ARTIFACTS", tmp_path)
    out = pf.fig_highdim_downstream()
    assert out == tmp_path / "fig_highdim_downstream.pdf"
    assert out.exists() and out.stat().st_size > 0


def test_fig_quotient_collapse_writes_pdf(highdim, tmp_path, monkeypatch):
    import analysis.paper_figures as pf

    monkeypatch.setattr(pf, "ARTIFACTS", tmp_path)
    out = pf.fig_quotient_collapse()
    assert out == tmp_path / "fig3_quotient_collapse.pdf"
    assert out.exists() and out.stat().st_size > 0


FACTORIAL = Path("results/factorial.parquet")

factorial_missing = pytest.mark.skipif(not FACTORIAL.exists(), reason="factorial parquet absent")


@pytest.fixture(scope="module")
def factorial() -> pd.DataFrame:
    return pd.read_parquet(FACTORIAL)


@factorial_missing
class TestPrepMeasureDependence:
    def test_orderings_rule1(self, factorial):
        from analysis.paper_figures import prep_measure_dependence

        prep = prep_measure_dependence(factorial)
        nonlin = prep[prep["dependence"] == "nonlinear"].set_index("relevance")["mean"]
        lin = prep[prep["dependence"] == "linear"].set_index("relevance")["mean"]
        # Rule 1: nonlinear-family measures (MI, dcor) beat linear-family measures on
        # nonlinear structure; Pearson tops linear structure and MI trails it there.
        assert nonlin.idxmax() == "mutual_info_regression"
        for winner in ["mutual_info_regression", "mutual_info_classif", "distance_corr"]:
            for loser in ["pearson_abs", "spearman_abs", "f_regression"]:
                assert nonlin[winner] > nonlin[loser] + 0.10
        assert lin.idxmax() == "pearson_abs"
        assert lin["pearson_abs"] > lin["mutual_info_classif"] + 0.10

    def test_anchor_values(self, factorial):
        from analysis.paper_figures import prep_measure_dependence

        prep = prep_measure_dependence(factorial)
        nonlin = prep[prep["dependence"] == "nonlinear"].set_index("relevance")["mean"]
        lin = prep[prep["dependence"] == "linear"].set_index("relevance")["mean"]
        # 10-seed values, verified against the committed factorial run artifact.
        assert nonlin["mutual_info_regression"] == pytest.approx(0.448, abs=0.01)
        assert nonlin["distance_corr"] == pytest.approx(0.407, abs=0.01)
        assert nonlin["mutual_info_classif"] == pytest.approx(0.393, abs=0.01)
        assert nonlin["pearson_abs"] == pytest.approx(0.177, abs=0.01)
        assert lin["pearson_abs"] == pytest.approx(0.575, abs=0.01)
        assert lin["mutual_info_classif"] == pytest.approx(0.409, abs=0.01)


@factorial_missing
def test_golden_is_fifteen_ground_truth_datasets(factorial):
    from analysis.paper_figures import golden

    g = golden(factorial)
    assert g["dataset"].nunique() == 15
    assert "friedman1" not in set(g["dataset"])


@factorial_missing
def test_fig_measure_dependence_writes_pdf(tmp_path, monkeypatch):
    import analysis.paper_figures as pf

    monkeypatch.setattr(pf, "ARTIFACTS", tmp_path)
    out = pf.fig_measure_dependence()
    assert out == tmp_path / "fig2_measure_dependence.pdf"
    assert out.exists() and out.stat().st_size > 0


@factorial_missing
class TestOperatorFigs:
    def test_lowdim_anchors(self, factorial):
        from analysis.paper_figures import prep_operator_lowdim

        prep = prep_operator_lowdim(factorial)
        f1 = prep[prep["metric"] == "f1"].set_index("operator")["mean"]
        noise = prep[prep["metric"] == "noise_rate"].set_index("operator")["mean"]
        # Honest low-dim story: F1s overlap within ~0.05; quotient clearly worst on noise.
        assert f1.max() - f1.min() < 0.06
        assert noise["quotient"] > noise["difference"] + 0.08
        assert noise["quotient"] > noise["multiplicative"] + 0.08
        assert f1["multiplicative"] == pytest.approx(0.468, abs=0.08)
        assert noise["quotient"] == pytest.approx(0.414, abs=0.08)

    def test_heatmap_prep_shape_and_quotient_column(self, factorial):
        from analysis.paper_figures import prep_operator_aggregation

        prep = prep_operator_aggregation(factorial)
        assert set(prep["metric"]) == {"f1", "noise_rate"}
        noise = prep[prep["metric"] == "noise_rate"].pivot(
            index="aggregation", columns="operator", values="mean"
        )
        # quotient is the most noise-prone cell in every aggregation row
        assert (noise["quotient"] >= noise[["difference", "multiplicative"]].max(axis=1)).all()


@factorial_missing
def test_fig4_and_fig5_write_pdfs(tmp_path, monkeypatch):
    import analysis.paper_figures as pf

    monkeypatch.setattr(pf, "ARTIFACTS", tmp_path)
    assert pf.fig_operator_lowdim().exists()
    assert pf.fig_operator_aggregation().exists()


@factorial_missing
class TestStopping:
    def test_anchors(self, factorial):
        from analysis.paper_figures import prep_stopping

        prep = prep_stopping(factorial)
        f1 = prep[prep["metric"] == "f1"].set_index("stop_mode")["mean"]
        size = prep[prep["metric"] == "k"].set_index("stop_mode")["mean"]
        # Rule 3: validation-selected fixed-k wins recovery at roughly half the size.
        assert f1.idxmax() == "val_fixed_k"
        assert f1["val_fixed_k"] == pytest.approx(0.519, abs=0.05)
        assert size["val_fixed_k"] == pytest.approx(4.41, abs=0.5)
        assert size["threshold"] > 1.5 * size["fixed_k"]

    def test_fig_writes(self, tmp_path, monkeypatch):
        import analysis.paper_figures as pf

        monkeypatch.setattr(pf, "ARTIFACTS", tmp_path)
        assert pf.fig_stopping().exists()


MI_WINNER = Path("results/mi_comparison_winner.csv")


@pytest.mark.skipif(not MI_WINNER.exists(), reason="mi comparison csv absent")
class TestMiWash:
    def test_wash_spread_and_pearson_gap(self):
        from analysis.paper_figures import PEARSON_LINEAR_AP, prep_mi_wash

        prep = prep_mi_wash()
        assert len(prep) == 12
        spread = prep["mean_ap_golden"].max() - prep["mean_ap_golden"].min()
        assert spread == pytest.approx(0.066, abs=0.01)  # the wash
        # no estimator within 0.05 of Pearson on linear structure
        assert (prep["mean_ap_linear"] < PEARSON_LINEAR_AP - 0.05).all()

    def test_incumbent_best_in_class_nonlinear(self):
        from analysis.paper_figures import prep_mi_wash

        prep = prep_mi_wash()
        best = prep.loc[prep["mean_ap_nonlinear"].idxmax(), "estimator"]
        assert best == "mi_reg_k3"

    def test_fig_writes(self, tmp_path, monkeypatch):
        import analysis.paper_figures as pf

        monkeypatch.setattr(pf, "ARTIFACTS", tmp_path)
        assert pf.fig_mi_wash().exists()
