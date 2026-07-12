"""Tests for the paper-artifact display-name mapping (fix-plan-v3 F1.1-display)."""

from analysis.display_names import display_spec, display_token


def test_display_token_maps_multiplicative_to_gate():
    assert display_token("multiplicative") == "gate"


def test_display_token_maps_reg_quotient():
    assert display_token("reg_quotient") == "reg. quotient"


def test_display_token_passthrough_for_unmapped():
    for token in ("difference", "quotient", "mean", "max", "sum", "dcor", "pearson_abs"):
        assert display_token(token) == token


def test_display_spec_maps_only_the_operator_segment():
    assert display_spec("dcor|pearson_abs|multiplicative|mean") == "dcor|pearson_abs|gate|mean"


def test_display_spec_passthrough_when_no_mapped_token():
    assert display_spec("f|pearson_abs|difference|mean") == "f|pearson_abs|difference|mean"
