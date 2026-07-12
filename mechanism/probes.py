"""Real-data ground truth via probe injection (artificial contrast variables,
Stoppiglia et al. 2003; Tuv et al. 2009). Append known-junk columns (shuffled
copies of real columns) whose relation to y is destroyed; a good selector rejects
them. Recovery on probe sets scores probe-rejection (the real columns' true
relevance is unknown)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mechanism.ground_truth import GroundTruth


def inject_probes(X, y, n_probes, seed):
    rng = np.random.default_rng(seed)
    X = X.reset_index(drop=True)
    p0 = X.shape[1]
    base_cols = rng.integers(0, p0, size=n_probes)
    probes = {}
    for j, src in enumerate(base_cols):
        col = X.iloc[:, int(src)].to_numpy()
        probes[f"probe_{j}"] = rng.permutation(col)  # shuffle destroys the y-relation
    Xp = pd.concat([X, pd.DataFrame(probes, index=X.index)], axis=1)
    gt = GroundTruth(
        informative=tuple(range(p0)),
        codependent=(),
        noise=tuple(range(p0, p0 + n_probes)),
        dependence="mixed",
        n_features=p0 + n_probes,
    )
    return Xp, gt
