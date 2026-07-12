import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from mechanism.probes import inject_probes


def test_probes_appended_and_recorded_as_noise():
    bunch = load_breast_cancer(as_frame=True)
    X, y = bunch.data.reset_index(drop=True), pd.Series(bunch.target)
    p0 = X.shape[1]
    Xp, gt = inject_probes(X, y, n_probes=10, seed=0)
    assert Xp.shape[1] == p0 + 10
    assert len(gt.noise) == 10
    assert set(gt.informative) == set(range(p0))  # originals are the "signal" side
    assert all(idx >= p0 for idx in gt.noise)


def test_probe_columns_have_destroyed_target_relation():
    from scipy.stats import pearsonr

    bunch = load_breast_cancer(as_frame=True)
    X, y = bunch.data.reset_index(drop=True), pd.Series(bunch.target)
    Xp, gt = inject_probes(X, y, n_probes=5, seed=1)
    # a probe's correlation with y should be near zero on average
    corrs = [abs(pearsonr(Xp.iloc[:, i].to_numpy(), y.to_numpy())[0]) for i in gt.noise]
    assert np.mean(corrs) < 0.2


def test_inject_probes_deterministic():
    bunch = load_breast_cancer(as_frame=True)
    X, y = bunch.data.reset_index(drop=True), pd.Series(bunch.target)
    a, ga = inject_probes(X, y, 8, seed=2)
    b, gb = inject_probes(X, y, 8, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert ga == gb
