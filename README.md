# modmrmr

**ModMRMR** is an injectable Maximum-Relevance Minimum-Redundancy (mRMR) feature
selection library. It provides a scikit-learn-compatible `MRMR` baseline and a
`ModMRMR` variant that multiplicatively suppresses redundancy against the
*already-selected* feature set (rather than averaging redundancy against all
selected features). Both relevance and redundancy scoring are injectable:
plug in linear correlation, mutual-information / adaptive-mutual-information
(AMI/KSG) estimators, or tree-based importances, or register your own scorer.
On top of the general estimators, `modmrmr` ships a time-aware `tmrmr` layer
(`run_tmrmr`) that performs forecast-safe, lag-aware covariate selection for
time series: it enforces `lag >= forecast_horizon + availability_margin` so
selected features can never leak future information into the forecast.

## Install

For development (clones the repo and installs the dev dependency group):

```bash
uv sync
```

To reproduce the paper's figures, tables, and benchmark studies, also install
the `benchmarks` dependency group (adds `matplotlib`, `pyarrow`, and the
baseline libraries the studies compare against):

```bash
uv sync --group benchmarks
```

Most benchmark datasets load from scikit-learn or OpenML on demand. Seven —
five scikit-feature microarray sets plus the two UCI archives — are downloaded
into `benchmarks/data/` on first use, checksum-verified against pinned sources.
To provision them up front (e.g. before an offline run):

```bash
uv run python -m benchmarks.fetch
```

As a dependency in another project:

```bash
uv add modmrmr
```

## Quickstart

### A. `ModMRMR` in an sklearn `Pipeline`

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from modmrmr import ModMRMR

rng = np.random.default_rng(0)
n = 200
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
x3 = rng.normal(size=n)
noise = rng.normal(size=n)
y = 3.0 * x1 - 2.0 * x2 + 0.1 * noise

X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x1_noisy": x1 + 0.01 * noise})

pipe = Pipeline(
    [
        ("select", ModMRMR(n_features=3, random_state=0)),
        ("lr", LinearRegression()),
    ]
)
pipe.fit(X, y)
predictions = pipe.predict(X)

print(pipe.named_steps["select"].selected_features_)
```

`ModMRMR` (and its `MRMR` baseline) is a standard `TransformerMixin`:
`importance_function` (relevance) and `penalty_function` (redundancy) are
constructor-injected callables, defaulting to `mutual_info_regression` and an
absolute Pearson correlation matrix respectively — swap in any callable with
a compatible signature to change the relevance or redundancy method.

### B. `MRMRSelector` — the configurable engine

`MRMRSelector` generalizes `MRMR`/`ModMRMR` into one estimator spanning the four
mRMR axes — **relevance**, **redundancy**, **operator**
(`difference`/`quotient`/`multiplicative`), and **aggregation**
(`mean`/`max`/`sum`) — with an automatic classification/regression path. Pass a
registered scorer name or a callable for `relevance`/`redundancy`; `"auto"`
resolves task-appropriate defaults (`f_classif`/`f_regression` relevance,
`pearson_abs` redundancy) via `detect_task`.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from modmrmr import MRMRSelector

X_arr, y = make_classification(
    n_samples=300, n_features=10, n_informative=4, n_redundant=3, random_state=0
)
X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])

# task="auto" detects classification and uses f_classif relevance + pearson_abs redundancy.
pipe = Pipeline(
    [
        ("select", MRMRSelector(n_features=4, operator="quotient", random_state=0)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
pipe.fit(X, y)
print(pipe.named_steps["select"].selected_features_)
print(pipe.named_steps["select"].task_)  # "classification"
```

The legacy presets are exact special cases: `MRMRSelector(operator="quotient",
aggregation="mean")` reproduces `MRMR`, and `MRMRSelector(operator="multiplicative",
aggregation="max")` reproduces `ModMRMR` (given the same relevance/redundancy
measures). Fitted selectors expose `selected_idx_`, `selected_features_`,
`selection_order_`, `selection_scores_`, and `task_`.

### C. `run_tmrmr` — forecast-safe, lag-aware covariate selection

```python
import numpy as np

from modmrmr import LagAwareModMRMRConfig, PairwiseScorerSpec, run_tmrmr

rng = np.random.default_rng(0)
n = 200
target = rng.normal(size=n).cumsum()
covariates = {
    "sensor_a": rng.normal(size=n).cumsum(),
    "sensor_b": rng.normal(size=n),
}

config = LagAwareModMRMRConfig(
    forecast_horizon=1,
    max_lag=3,
    max_selected_features=2,
    relevance_scorer=PairwiseScorerSpec(
        name="pearson_abs",
        backend="scipy",
        normalization="rank_percentile",
        significance_method="none",
    ),
    redundancy_scorer=PairwiseScorerSpec(
        name="pearson_abs",
        backend="scipy",
        normalization="rank_percentile",
        significance_method="none",
    ),
)

result = run_tmrmr(target=target, covariates=covariates, config=config, random_state=0)
print([f.feature_name for f in result.selected])
```

`run_tmrmr` only ever evaluates lags that satisfy
`lag >= forecast_horizon + availability_margin`; anything shorter is recorded
in `result.blocked` rather than scored, so the selected set is always safe to
use at inference time. `relevance_scorer` and `redundancy_scorer` (and the
optional `target_history_scorer`) each accept any registered scorer name via
`PairwiseScorerSpec`, so relevance and redundancy can use different methods
(e.g. linear relevance with mutual-information redundancy).

### Injectable scorers

```python
from modmrmr import list_scorers, register_scorer

print(list_scorers())
# ['catt_knn_mi', 'cross_ami_score', 'distance_corr', 'f_classif',
#  'f_regression', 'gcmi', 'ksg_mi', 'mutual_info_classif',
#  'mutual_info_sklearn', 'pearson_abs', 'rdc', 'relieff',
#  'spearman_abs', 'tree_r2']

# Register a custom scorer (must implement `score_pair(x, y, *, random_state)`):
# register_scorer("my_scorer", MyScorer())
```
