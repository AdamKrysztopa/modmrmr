"""Task detection heuristic + default relevance/redundancy measures per task.

Fixes the legacy regression-only defect: the old ``MRMR`` hardcoded
``mutual_info_regression`` and cast ``y`` to float, silently mangling
classification targets. :func:`detect_task` routes to the right defaults.
"""

from __future__ import annotations

import numpy as np

# task -> (relevance_name, redundancy_name). Names resolve via the scorer
# registry (redundancy) or sklearn.feature_selection (relevance) in the selector.
DEFAULT_MEASURES: dict[str, tuple[str, str]] = {
    "classification": ("f_classif", "pearson_abs"),
    "regression": ("f_regression", "pearson_abs"),
}


def detect_task(y) -> str:
    """Infer ``"classification"`` or ``"regression"`` from a target vector.

    Heuristic (per the interface contract): a non-numeric dtype, OR an integer
    dtype whose number of unique values is ``<= max(20, 0.05 * n)``, is treated
    as classification. Everything else (continuous floats, high-cardinality
    integers) is regression. ``bool`` is non-numeric under numpy and therefore
    classification.

    Args:
        y: array-like target (numpy array, pandas Series, or list).

    Returns:
        ``"classification"`` or ``"regression"``.
    """
    values = np.asarray(y)
    dtype = values.dtype

    # Non-numeric (strings, object, categorical, bool) -> classification.
    if not np.issubdtype(dtype, np.number):
        return "classification"

    n = values.shape[0]
    if np.issubdtype(dtype, np.integer):
        n_unique = np.unique(values).shape[0]
        if n_unique <= max(20, int(0.05 * n)):
            return "classification"

    return "regression"
