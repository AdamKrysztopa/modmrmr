"""Shared publication figure style: Okabe-Ito palette with fixed semantic assignments,
paper rcParams, and seed-deterministic bootstrap CIs (design spec §4)."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

OPERATOR_COLORS = {
    "difference": OKABE_ITO["blue"],
    "multiplicative": OKABE_ITO["green"],
    "quotient": OKABE_ITO["vermillion"],
    "mult_max": OKABE_ITO["purple"],
}

MEASURE_COLORS = {
    "linear": OKABE_ITO["blue"],
    "f_test": OKABE_ITO["sky"],
    "mutual_info": OKABE_ITO["orange"],
    "distance_corr": OKABE_ITO["green"],
    "model_based": OKABE_ITO["purple"],
}

_PAPER_RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
}


@contextmanager
def paper_rc():
    import matplotlib

    with matplotlib.rc_context(_PAPER_RC):
        yield


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 10000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        v = float(arr[0]) if arr.size else float("nan")
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))
