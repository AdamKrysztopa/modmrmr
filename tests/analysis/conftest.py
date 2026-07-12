import pandas as pd
import pytest

from analysis.schema import make_synthetic_results


@pytest.fixture(scope="session")
def synthetic_results() -> pd.DataFrame:
    return make_synthetic_results()
