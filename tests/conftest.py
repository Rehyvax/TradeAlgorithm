"""Shared fixtures: synthetic price panels, so no test needs the network."""

import numpy as np
import pandas as pd
import pytest

SLEEVES_TEST = {
    "equity": ["EQ1", "EQ2", "EQ3"],
    "bonds": ["BD1", "BD2"],
}


def make_prices(n_days: int = 700, seed: int = 0, drifts=None) -> pd.DataFrame:
    """Geometric random walks for five risky assets and a cash ETF."""
    rng = np.random.default_rng(seed)
    tickers = SLEEVES_TEST["equity"] + SLEEVES_TEST["bonds"]
    drifts = drifts or {"EQ1": 0.0006, "EQ2": 0.0002, "EQ3": -0.0004, "BD1": 0.0002, "BD2": -0.0001}
    vols = {"EQ1": 0.012, "EQ2": 0.012, "EQ3": 0.014, "BD1": 0.004, "BD2": 0.005}
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    data = {}
    for t in tickers:
        r = drifts[t] + vols[t] * rng.standard_normal(n_days)
        data[t] = 100.0 * np.exp(np.cumsum(r))
    data["SHY"] = 100.0 * np.exp(np.cumsum(np.full(n_days, 0.00008)))
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def sleeves() -> dict:
    return SLEEVES_TEST
