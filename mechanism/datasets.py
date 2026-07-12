"""Dataset registry for the mechanism suite.

Wires the seeded synthetic generators (``mechanism.synthetic``) and two
probe-injected real datasets (``mechanism.probes``) behind a single
name -> loader registry, mirroring ``benchmarks.datasets`` but carrying a
``GroundTruth`` mask instead of a bare ``(X, y)`` pair.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from mechanism.ground_truth import GroundTruth
from mechanism.probes import inject_probes
from mechanism.synthetic import (
    make_annulus,
    make_checkerboard,
    make_conditional_redundancy,
    make_heteroscedastic,
    make_linear_control_clf,
    make_linear_control_reg,
    make_max_of,
    make_mixed,
    make_multiplicative_interaction,
    make_nonlin_redundancy_clf,
    make_nonlin_redundancy_reg,
    make_parabola,
    make_polynomial_sign,
    make_quotient_trap_clf,
    make_quotient_trap_reg,
    make_radial,
    make_ratio,
    make_redundant_blocks_clf,
    make_redundant_blocks_reg,
    make_saturation,
    make_sinc_2d,
    make_sine,
    make_threshold_and,
    make_xor,
)

_PROBE_N = 10


def _load_probe_breast_cancer() -> tuple[pd.DataFrame, pd.Series, GroundTruth]:
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.reset_index(drop=True)
    y = pd.Series(bunch.target).reset_index(drop=True)
    Xp, gt = inject_probes(X, y, n_probes=_PROBE_N, seed=0)
    return Xp, y, gt


def _load_probe_diabetes() -> tuple[pd.DataFrame, pd.Series, GroundTruth]:
    bunch = load_diabetes(as_frame=True)
    X = bunch.data.reset_index(drop=True)
    y = pd.Series(bunch.target).reset_index(drop=True)
    Xp, gt = inject_probes(X, y, n_probes=_PROBE_N, seed=0)
    return Xp, y, gt


MECHANISM_DATASETS: dict[str, dict] = {
    "parabola": {"loader": make_parabola, "task": "regression", "dependence": "nonlinear"},
    "radial": {"loader": make_radial, "task": "classification", "dependence": "nonlinear"},
    "sine": {"loader": make_sine, "task": "regression", "dependence": "nonlinear"},
    "nonlin_redundancy_clf": {
        "loader": make_nonlin_redundancy_clf,
        "task": "classification",
        "dependence": "mixed",
    },
    "nonlin_redundancy_reg": {
        "loader": make_nonlin_redundancy_reg,
        "task": "regression",
        "dependence": "mixed",
    },
    "linear_control_clf": {
        "loader": make_linear_control_clf,
        "task": "classification",
        "dependence": "linear",
    },
    "linear_control_reg": {
        "loader": make_linear_control_reg,
        "task": "regression",
        "dependence": "linear",
    },
    "mixed": {"loader": make_mixed, "task": "classification", "dependence": "mixed"},
    "xor": {"loader": make_xor, "task": "classification", "dependence": "nonlinear"},
    "checkerboard": {
        "loader": make_checkerboard,
        "task": "classification",
        "dependence": "nonlinear",
    },
    "annulus": {"loader": make_annulus, "task": "classification", "dependence": "nonlinear"},
    "multiplicative_interaction": {
        "loader": make_multiplicative_interaction,
        "task": "regression",
        "dependence": "nonlinear",
    },
    "threshold_and": {
        "loader": make_threshold_and,
        "task": "classification",
        "dependence": "nonlinear",
    },
    "saturation": {"loader": make_saturation, "task": "regression", "dependence": "nonlinear"},
    "ratio": {"loader": make_ratio, "task": "regression", "dependence": "nonlinear"},
    "max_of": {"loader": make_max_of, "task": "regression", "dependence": "nonlinear"},
    "heteroscedastic": {
        "loader": make_heteroscedastic,
        "task": "regression",
        "dependence": "nonlinear",
    },
    "sinc_2d": {"loader": make_sinc_2d, "task": "regression", "dependence": "nonlinear"},
    "polynomial_sign": {
        "loader": make_polynomial_sign,
        "task": "classification",
        "dependence": "nonlinear",
    },
    "conditional_redundancy": {
        "loader": make_conditional_redundancy,
        "task": "classification",
        "dependence": "nonlinear",
    },
    "redundant_blocks_clf": {
        "loader": make_redundant_blocks_clf,
        "task": "classification",
        "dependence": "linear",
    },
    "redundant_blocks_reg": {
        "loader": make_redundant_blocks_reg,
        "task": "regression",
        "dependence": "linear",
    },
    "quotient_trap_reg": {
        "loader": make_quotient_trap_reg,
        "task": "regression",
        "dependence": "linear",
    },
    "quotient_trap_clf": {
        "loader": make_quotient_trap_clf,
        "task": "classification",
        "dependence": "linear",
    },
    "probe_breast_cancer": {
        "loader": _load_probe_breast_cancer,
        "task": "classification",
        "dependence": "mixed",
    },
    "probe_diabetes": {
        "loader": _load_probe_diabetes,
        "task": "regression",
        "dependence": "mixed",
    },
}


def list_mechanism_datasets() -> list[str]:
    """Return the registered mechanism dataset names."""
    return list(MECHANISM_DATASETS)


def load_mechanism_dataset(name: str) -> tuple[pd.DataFrame, pd.Series, str, GroundTruth]:
    """Load a mechanism dataset by name -> ``(X, y, task, gt)``.

    Raises ``KeyError`` if ``name`` is not registered.
    """
    try:
        entry = MECHANISM_DATASETS[name]
    except KeyError:
        valid = ", ".join(sorted(MECHANISM_DATASETS))
        raise KeyError(f"unknown mechanism dataset {name!r}; valid names: {valid}") from None
    X, y, gt = entry["loader"]()
    return X, y, entry["task"], gt
