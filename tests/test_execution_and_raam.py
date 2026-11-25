"""Tests for the order cash budget and the Ranked Asset Allocation Model."""

import numpy as np
import pandas as pd
import pytest

from alpaca.trading.enums import OrderSide

import raam_replication as raam
from daily_execution import apply_cash_budget


def _order(symbol, side, notional):
    return {"symbol": symbol, "side": side, "notional": notional}


def test_cash_budget_scales_buys_and_keeps_sells():
    orders = [_order("A", OrderSide.BUY, 600.0), _order("B", OrderSide.BUY, 400.0), _order("C", OrderSide.SELL, 300.0)]
    out = apply_cash_budget(orders, cash_available=600.0, cash_buffer_usd=100.0)
    assert str(out[0]["side"]) == str(OrderSide.SELL)
    buys = [o for o in out if str(o["side"]) == str(OrderSide.BUY)]
    assert sum(o["notional"] for o in buys) == pytest.approx(500.0)
    assert buys[0]["notional"] / buys[1]["notional"] == pytest.approx(1.5)


def test_cash_budget_leaves_affordable_orders_untouched():
    orders = [_order("A", OrderSide.BUY, 100.0)]
    out = apply_cash_budget(orders, cash_available=1000.0, cash_buffer_usd=100.0)
    assert out[0]["notional"] == 100.0


@pytest.fixture
def raam_prices():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2016-01-01", periods=600)
    risky = [f"R{i}" for i in range(8)]
    drifts = np.linspace(-0.0008, 0.0010, len(risky))
    data = {t: 100 * np.exp(np.cumsum(d + 0.01 * rng.standard_normal(len(idx)))) for t, d in zip(risky, drifts)}
    data["SHY"] = 100 * np.exp(np.cumsum(np.full(len(idx), 0.00008)))
    return pd.DataFrame(data, index=idx), risky


def test_raam_holds_n_slots_and_sums_to_one(raam_prices):
    prices, risky = raam_prices
    w = raam.raam_weights(prices, raam.RaamConfig(), risky)
    live = w[w.sum(axis=1) > 0]
    np.testing.assert_allclose(live.sum(axis=1), 1.0)
    assert ((live.drop(columns="SHY") > 0).sum(axis=1) <= 5).all()


def test_raam_sends_negative_momentum_to_cash(raam_prices):
    prices, risky = raam_prices
    cfg = raam.RaamConfig()
    w = raam.raam_weights(prices, cfg, risky)
    mom = prices[risky] / prices[risky].shift(cfg.momentum_window) - 1.0
    for t in raam.month_ends(prices.index):
        if w.loc[t].sum() == 0:
            continue
        held = w.loc[t].drop("SHY")
        assert (mom.loc[t, held[held > 0].index] > 0).all()


def test_raam_latest_matches_full_history(raam_prices):
    prices, risky = raam_prices
    full = raam.raam_weights(prices, raam.RaamConfig(), risky)
    last_me = raam.month_ends(prices.index)[-1]
    latest = raam.latest_raam_weights(prices, raam.RaamConfig(), risky, history=400)
    expected = full.loc[prices.index[-1]]
    pd.testing.assert_series_equal(latest.sort_index(), expected[expected > 0].sort_index(), check_names=False)
    assert last_me <= prices.index[-1]
