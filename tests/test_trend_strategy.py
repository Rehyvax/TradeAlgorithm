"""Tests for the core trend strategy."""

from dataclasses import replace

import numpy as np
import pandas as pd

from trend_strategy import (
    TrendConfig,
    compute_trend_score,
    latest_target_weights,
    run_trend_strategy,
)

CFG = TrendConfig(warmup=260)


def test_trend_score_bounds_and_warmup(prices):
    score = compute_trend_score(prices, (63, 126, 252), long_only=True)
    assert score.iloc[:252].isna().all().all()
    valid = score.iloc[252:]
    assert ((valid >= 0) & (valid <= 1)).all().all()


def test_trend_score_long_short_range(prices):
    score = compute_trend_score(prices, (63, 126, 252), long_only=False).iloc[252:]
    assert ((score >= -1) & (score <= 1)).all().all()


def test_weights_are_long_only_and_fully_allocated(prices, sleeves):
    w = run_trend_strategy(prices, CFG, sleeves)["weights"].iloc[CFG.warmup + 5:]
    assert (w >= -1e-12).all().all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_single_asset_cap(prices, sleeves):
    w = run_trend_strategy(prices, CFG, sleeves)["weights"].drop(columns="SHY")
    assert w.max().max() <= CFG.max_weight + 1e-9


def test_no_look_ahead(prices, sleeves):
    """Changing prices after a date must not change any weight up to that date."""
    cut = 500
    base = run_trend_strategy(prices, CFG, sleeves)["weights"]
    shocked = prices.copy()
    shocked.iloc[cut + 1:] *= np.linspace(1.0, 3.0, len(prices) - cut - 1)[:, None]
    alt = run_trend_strategy(shocked, CFG, sleeves)["weights"]
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], alt.iloc[: cut + 1])


def test_returns_use_previous_day_weights(prices, sleeves):
    res = run_trend_strategy(prices, replace(CFG, cost_bps=0.0), sleeves)
    expected = (res["weights"].shift(1).fillna(0.0) * prices.pct_change().fillna(0.0)).sum(axis=1)
    pd.testing.assert_series_equal(res["returns"], expected, check_names=False)


def test_costs_reduce_returns(prices, sleeves):
    cheap = run_trend_strategy(prices, replace(CFG, cost_bps=0.0), sleeves)["returns"].sum()
    dear = run_trend_strategy(prices, replace(CFG, cost_bps=50.0), sleeves)["returns"].sum()
    assert dear < cheap


def test_downtrending_asset_is_not_held(sleeves):
    from tests.conftest import make_prices

    p = make_prices(drifts={"EQ1": 0.0005, "EQ2": 0.0005, "EQ3": -0.004, "BD1": 0.0002, "BD2": 0.0002})
    w = run_trend_strategy(p, CFG, sleeves)["weights"]
    assert w["EQ3"].iloc[400:].max() < 1e-9


def test_latest_weights_are_valid(prices, sleeves):
    w = latest_target_weights(prices, CFG, sleeves)
    assert (w >= -1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-9
