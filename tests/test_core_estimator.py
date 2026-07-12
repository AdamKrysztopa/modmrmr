import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from modmrmr.core import MRMR, ModMRMR


def _dataset(n: int = 600):
    """Two independent signal groups; group A has a near-duplicate feature.

    y depends on both signals, so ``relevant_a`` and ``relevant_b`` are both
    informative while ``duplicate_a`` is a near-copy of ``relevant_a``. With a
    2-feature budget, a redundancy-aware selector should keep one A-feature
    plus ``relevant_b``, dropping the duplicate.
    """
    rng = np.random.default_rng(0)
    signal_a = rng.normal(size=n)
    signal_b = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "relevant_a": signal_a + rng.normal(scale=0.1, size=n),
            "duplicate_a": signal_a + rng.normal(scale=0.1, size=n),
            "relevant_b": signal_b + rng.normal(scale=0.1, size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = signal_a + signal_b
    return X, y


def test_modmrmr_drops_near_duplicate_for_diverse_feature() -> None:
    X, y = _dataset()
    picks = set(ModMRMR(n_features=2, random_state=0).fit(X, y).selected_features_)
    # The independent signal must be represented...
    assert "relevant_b" in picks
    # ...and the two A near-duplicates must not both be selected.
    assert not {"relevant_a", "duplicate_a"}.issubset(picks)


def test_transform_preserves_input_type_and_width() -> None:
    X, y = _dataset()
    model = ModMRMR(n_features=2, random_state=0).fit(X, y)
    assert model.transform(X).shape[1] == 2
    assert isinstance(model.transform(X.to_numpy()), np.ndarray)


def test_clone_and_pipeline_compatible() -> None:
    X, y = _dataset()
    cloned = clone(ModMRMR(n_features=2, random_state=0))
    pipe = Pipeline([("select", cloned), ("lr", LinearRegression())]).fit(X, y)
    assert pipe.predict(X).shape[0] == X.shape[0]


def test_mrmr_and_modmrmr_score_formulas_rank_differently() -> None:
    """The quotient (MRMR) and product-penalty (ModMRMR) forms can rank the
    same candidates differently against an identical selected set."""
    nom = pd.Series({"c1": 1.0, "c2": 0.6, "s": 9.9})
    denom = pd.DataFrame(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.2], [0.5, 0.2, 1.0]],
        index=["c1", "c2", "s"],
        columns=["c1", "c2", "s"],
    )
    selected, not_selected = ["s"], ["c1", "c2"]
    mrmr_pick = MRMR(random_state=0)._get_scores(nom, denom, selected, not_selected).idxmax()
    modmrmr_pick = ModMRMR(random_state=0)._get_scores(nom, denom, selected, not_selected).idxmax()
    # MRMR: c1=1.0/0.5=2.0, c2=0.6/0.2=3.0 -> c2.
    # ModMRMR: c1=1.0*(1-0.5)=0.5, c2=0.6*(1-0.2)=0.48 -> c1.
    assert mrmr_pick == "c2"
    assert modmrmr_pick == "c1"
    assert mrmr_pick != modmrmr_pick
