import re

import numpy as np
import pytest

from mechanism.paper_style import (
    MEASURE_COLORS,
    OKABE_ITO,
    OPERATOR_COLORS,
    bootstrap_ci,
    paper_rc,
)

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_palettes_are_valid_distinct_hex():
    for palette in (OKABE_ITO, OPERATOR_COLORS, MEASURE_COLORS):
        assert all(_HEX.match(c) for c in palette.values())
        assert len(set(palette.values())) == len(palette)


def test_operator_keys_match_study_vocabulary():
    assert set(OPERATOR_COLORS) == {"difference", "multiplicative", "quotient", "mult_max"}


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=30)
    lo, hi = bootstrap_ci(values)
    assert lo < float(np.mean(values)) < hi
    assert (lo, hi) == bootstrap_ci(values)


def test_bootstrap_ci_degenerate_single_value():
    lo, hi = bootstrap_ci([0.7])
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)


def test_paper_rc_sets_and_restores():
    import matplotlib

    before = matplotlib.rcParams["font.size"]
    with paper_rc():
        assert matplotlib.rcParams["font.size"] >= 8
    assert matplotlib.rcParams["font.size"] == before
