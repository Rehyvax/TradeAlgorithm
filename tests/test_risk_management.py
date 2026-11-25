"""Tests for the account-level risk overlay."""

import numpy as np
import pandas as pd
import pytest

from risk_management import (
    RiskConfig,
    compute_exposure,
    current_drawdown,
    drawdown_multiplier,
    replay_kill_switch,
    sanitize_equity_history,
    volatility_multiplier,
)

CFG = RiskConfig()


def _eq(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


@pytest.mark.parametrize(
    "dd, expected",
    [(0.0, 1.0), (CFG.dd_soft, 1.0), (CFG.dd_hard, CFG.dd_floor), (CFG.dd_kill, 0.0), (-0.50, 0.0)],
)
def test_drawdown_multiplier_anchor_points(dd, expected):
    assert drawdown_multiplier(dd, CFG) == pytest.approx(expected)


def test_drawdown_multiplier_is_monotonic():
    grid = np.linspace(0.0, -0.19, 50)
    values = [drawdown_multiplier(d, CFG) for d in grid]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_current_drawdown():
    assert current_drawdown(_eq([100, 120, 90])) == pytest.approx(-0.25)
    assert current_drawdown(_eq([100, 110, 120])) == 0.0


def test_sanitize_removes_spike_that_reverts():
    clean = sanitize_equity_history(_eq([100, 101, 70, 101, 102]))
    assert 70 not in clean.values
    assert len(clean) == 4


def test_sanitize_keeps_confirmed_move():
    clean = sanitize_equity_history(_eq([100, 101, 80, 79, 78]))
    assert len(clean) == 5


def test_kill_switch_fires_and_rearms():
    cfg = RiskConfig(kill_cooldown=5)
    path = [100.0] * 10 + [75.0] + [75.0] * 3
    assert replay_kill_switch(_eq(path), cfg)["active"]
    path += [75.0] * 5
    state = replay_kill_switch(_eq(path), cfg)
    assert not state["active"] and state["reentry"]


def test_volatility_multiplier_scales_down_high_vol():
    rng = np.random.default_rng(0)
    calm = pd.Series(0.002 * rng.standard_normal(300))
    wild = pd.Series(0.03 * rng.standard_normal(300))
    assert volatility_multiplier(calm, CFG) == pytest.approx(CFG.max_exposure)
    assert volatility_multiplier(wild, CFG) < 0.3


def test_compute_exposure_flags_unconfirmed_jump():
    out = compute_exposure(_eq([100, 101, 102, 60]))
    assert out["suspect_print"]


def test_compute_exposure_bounds():
    rng = np.random.default_rng(1)
    eq = _eq(100 * np.cumprod(1 + 0.004 * rng.standard_normal(400)))
    out = compute_exposure(eq)
    assert 0.0 <= out["exposure"] <= CFG.max_exposure
