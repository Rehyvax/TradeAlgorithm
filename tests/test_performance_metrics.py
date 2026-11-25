"""Tests for the reference performance metrics."""

import numpy as np
import pandas as pd
import pytest

from performance_metrics import (
    bootstrap_sharpe_interval,
    compute_performance_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=2520)
    return pd.Series(0.0004 + 0.01 * rng.standard_normal(2520), index=idx)


def test_sharpe_matches_definition(returns):
    m = compute_performance_metrics(returns)
    assert m["sharpe_ratio"] == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(252), rel=1e-6)


def test_max_drawdown_is_negative_and_bounded(returns):
    m = compute_performance_metrics(returns)
    assert -1.0 <= m["max_drawdown"] < 0.0


def test_psr_falls_as_benchmark_rises(returns):
    assert probabilistic_sharpe_ratio(returns, 0.0) > probabilistic_sharpe_ratio(returns, 1.0)


def test_expected_max_sharpe_grows_with_trials():
    vals = [expected_max_sharpe(n, 0.3) for n in (2, 10, 100)]
    assert vals == sorted(vals)


def test_dsr_below_psr_when_many_trials(returns):
    psr = probabilistic_sharpe_ratio(returns, 0.0)
    dsr = deflated_sharpe_ratio(returns, np.linspace(0.0, 0.8, 50))
    assert dsr < psr


def test_bootstrap_interval_contains_estimate(returns):
    lo, hi = bootstrap_sharpe_interval(returns, n_boot=300)
    sr = compute_performance_metrics(returns)["sharpe_ratio"]
    assert lo < sr < hi
