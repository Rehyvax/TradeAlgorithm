"""Portfolio-level risk controls for live execution.

The first version of ``daily_execution.py`` had two weaknesses that showed
up in the live paper account:

1. The drawdown filter used the *worst drawdown ever recorded* in
   ``equity_history.csv``.  Once the threshold had been crossed a single
   time, the system stayed halted for good, while the positions it already
   held kept drifting.  A stop that never releases and never reduces risk
   is not a risk control.
2. Account equity was trusted blindly.  One spurious print (an equity
   value far below the neighbouring days that reverted on the next run)
   was enough to trigger the permanent halt.

This module replaces both with rules that are standard in systematic
funds:

* equity prints are sanitised before any statistic is computed;
* the drawdown is measured *now*, against the peak of a rolling window;
* exposure is cut gradually as the drawdown deepens and restored as the
  account recovers, so there is always a path back to full risk;
* a hard kill-switch still exists, but it liquidates to cash and re-arms
  itself after a cooling-off period, re-entering at reduced exposure.

The functions are pure (no I/O) so that they can be unit-tested and reused
in backtests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    """Parameters of the live risk overlay.

    Parameters
    ----------
    target_vol : float
        Annualised volatility target for the whole account.
    vol_window : int
        Number of daily observations used to estimate realised volatility.
    max_exposure : float
        Upper bound for the exposure multiplier (1.0 means no leverage).
    dd_soft : float
        Drawdown (negative) at which exposure starts to be reduced.
    dd_hard : float
        Drawdown (negative) at which exposure reaches ``dd_floor``.
    dd_floor : float
        Exposure multiplier applied at or beyond ``dd_hard``.
    dd_kill : float
        Drawdown (negative) at which the account is moved fully to cash.
    dd_lookback : int
        Window, in observations, used to locate the reference peak.
    kill_cooldown : int
        Number of observations the kill-switch stays active before the
        account is allowed to re-enter at ``dd_floor`` exposure.
    max_equity_jump : float
        Absolute one-day equity change above which a print is treated as
        suspect.
    """

    target_vol: float = 0.10
    vol_window: int = 63
    max_exposure: float = 1.0
    dd_soft: float = -0.05
    dd_hard: float = -0.12
    dd_floor: float = 0.35
    dd_kill: float = -0.20
    dd_lookback: int = 252
    kill_cooldown: int = 21
    max_equity_jump: float = 0.15


def sanitize_equity_history(equity: pd.Series, max_jump: float = 0.15) -> pd.Series:
    """Remove spurious equity prints.

    A print is considered spurious when the account value moves by more
    than ``max_jump`` in one observation and the following observation
    undoes at least half of that move.  A diversified, unlevered ETF
    portfolio cannot lose 20 % overnight and regain it the next day, so
    such a pattern is a data problem (a partial account snapshot, a
    pending transfer), not a market event.

    Parameters
    ----------
    equity : pandas.Series
        Account equity ordered in time.
    max_jump : float, optional
        Threshold for a suspect one-observation change.

    Returns
    -------
    pandas.Series
        The series without the spurious observations.  The index of the
        surviving observations is preserved.
    """
    eq = equity.astype(float).copy()
    keep = np.ones(len(eq), dtype=bool)
    values = eq.to_numpy()
    last_good = values[0] if len(values) else np.nan
    for i in range(1, len(values)):
        change = values[i] / last_good - 1.0
        if abs(change) > max_jump and i + 1 < len(values):
            rebound = values[i + 1] / values[i] - 1.0
            if np.sign(rebound) == -np.sign(change) and abs(values[i + 1] / last_good - 1.0) < abs(change) / 2.0:
                keep[i] = False
                continue
        last_good = values[i]
    return eq[keep]


def is_suspect_print(previous_equity: float, current_equity: float, max_jump: float = 0.15) -> bool:
    """Flag today's equity when it differs too much from the last good print.

    Today's value cannot be confirmed by a later observation, so the safe
    action is to skip trading for the day and look again on the next run.
    """
    if not np.isfinite(previous_equity) or previous_equity <= 0:
        return False
    return abs(current_equity / previous_equity - 1.0) > max_jump


def current_drawdown(equity: pd.Series, lookback: int = 252) -> float:
    """Drawdown of the latest observation against the rolling peak.

    Parameters
    ----------
    equity : pandas.Series
        Sanitised account equity ordered in time.
    lookback : int, optional
        Number of trailing observations in which the peak is searched.

    Returns
    -------
    float
        A number in [-1, 0].  Zero means the account is at its peak.
    """
    if equity.empty:
        return 0.0
    window = equity.iloc[-lookback:]
    return float(window.iloc[-1] / window.max() - 1.0)


def drawdown_multiplier(drawdown: float, cfg: RiskConfig) -> float:
    """Exposure multiplier as a function of the current drawdown.

    The multiplier is 1.0 above ``dd_soft``, falls linearly to
    ``dd_floor`` at ``dd_hard`` and is 0.0 at or beyond ``dd_kill``.
    Because it depends only on the *current* drawdown it rises again as
    the account recovers.
    """
    if drawdown <= cfg.dd_kill:
        return 0.0
    if drawdown >= cfg.dd_soft:
        return 1.0
    if drawdown <= cfg.dd_hard:
        return cfg.dd_floor
    frac = (drawdown - cfg.dd_hard) / (cfg.dd_soft - cfg.dd_hard)
    return float(cfg.dd_floor + (1.0 - cfg.dd_floor) * frac)


def volatility_multiplier(daily_returns: pd.Series, cfg: RiskConfig) -> float:
    """Exposure multiplier that brings realised volatility to the target.

    Returns ``min(max_exposure, target_vol / realised_vol)``.  With fewer
    than ``vol_window`` observations the multiplier is 1.0.
    """
    r = daily_returns.dropna()
    if len(r) < cfg.vol_window:
        return 1.0
    realised = float(r.iloc[-cfg.vol_window:].std() * np.sqrt(252))
    if not np.isfinite(realised) or realised <= 0:
        return 1.0
    return float(min(cfg.max_exposure, cfg.target_vol / realised))


def replay_kill_switch(equity: pd.Series, cfg: RiskConfig) -> dict:
    """Replay the kill-switch over the equity history.

    The switch fires when the rolling drawdown reaches ``dd_kill``.  It
    then keeps the account in cash for ``kill_cooldown`` observations.
    When the cooling-off period ends the switch re-arms: the reference
    peak is reset to the current equity, so that an account sitting in
    cash is not locked out until the old peak leaves the window, and the
    exposure is capped at ``dd_floor`` for another ``kill_cooldown``
    observations.

    Parameters
    ----------
    equity : pandas.Series
        Sanitised account equity ordered in time.
    cfg : RiskConfig
        Risk parameters.

    Returns
    -------
    dict
        ``active`` (the account must be in cash), ``reentry`` (exposure
        is capped at ``dd_floor``) and ``drawdown`` (current drawdown
        against the reference peak).
    """
    values = equity.to_numpy(dtype=float)
    start, cooldown_left, reentry_left, dd = 0, 0, 0, 0.0
    for i in range(len(values)):
        lo = max(start, i - cfg.dd_lookback + 1)
        dd = values[i] / values[lo: i + 1].max() - 1.0
        if cooldown_left > 0:
            cooldown_left -= 1
            if cooldown_left == 0:
                start, reentry_left, dd = i, cfg.kill_cooldown, 0.0
        elif dd <= cfg.dd_kill:
            cooldown_left = cfg.kill_cooldown
        elif reentry_left > 0:
            reentry_left -= 1
    return {"active": cooldown_left > 0, "reentry": reentry_left > 0, "drawdown": float(dd)}


def compute_exposure(equity: pd.Series, cfg: RiskConfig | None = None) -> dict:
    """Combine all risk rules into one exposure multiplier.

    Parameters
    ----------
    equity : pandas.Series
        Raw account equity ordered in time, including today's value as the
        last observation.
    cfg : RiskConfig, optional
        Risk parameters.  Defaults to :class:`RiskConfig`.

    Returns
    -------
    dict
        ``exposure`` (final multiplier), ``drawdown``, ``dd_multiplier``,
        ``vol_multiplier``, ``kill_switch`` and ``suspect_print``.  When
        ``suspect_print`` is True the caller should not trade today.
    """
    cfg = cfg or RiskConfig()
    eq = equity.astype(float).dropna()
    out = {
        "exposure": cfg.max_exposure, "drawdown": 0.0, "dd_multiplier": 1.0,
        "vol_multiplier": 1.0, "kill_switch": False, "suspect_print": False,
    }
    if len(eq) < 2:
        return out
    clean = sanitize_equity_history(eq, cfg.max_equity_jump)
    today = float(eq.iloc[-1])
    if len(clean) >= 2:
        # Today's print is suspect only if it is far from the last good
        # print *and* from yesterday's raw print.  A large move that is
        # confirmed by a second observation is treated as real.
        far_from_good = is_suspect_print(float(clean.iloc[-2]), today, cfg.max_equity_jump)
        far_from_raw = is_suspect_print(float(eq.iloc[-2]), today, cfg.max_equity_jump)
        if far_from_good and far_from_raw:
            out["suspect_print"] = True
            return out
    state = replay_kill_switch(clean, cfg)
    dd = state["drawdown"]
    dd_mult = drawdown_multiplier(dd, cfg)
    if state["reentry"]:
        dd_mult = min(dd_mult, cfg.dd_floor)
    vol_mult = volatility_multiplier(clean.pct_change(), cfg)
    kill = state["active"]
    exposure = 0.0 if kill else min(cfg.max_exposure, dd_mult * vol_mult)
    out.update({
        "exposure": float(exposure), "drawdown": float(dd), "dd_multiplier": float(dd_mult),
        "vol_multiplier": float(vol_mult), "kill_switch": kill,
    })
    return out
