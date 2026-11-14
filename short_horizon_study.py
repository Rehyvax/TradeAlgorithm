"""
short_horizon_study.py
======================

Does momentum work over windows of a few days in semiconductor and AI
stocks?

Motivation
----------
Semiconductor names move in bursts: rallies of two or three days followed
by corrections of similar length.  Looking back at a chart it seems that
buying after a strong day and selling two days later, or stepping aside
after the first weak day, would have captured most of the upside and
avoided most of the downside.  This script tests that idea on data
instead of on memory.

What is tested
--------------
For each ticker and each evaluation period:

1. **Persistence statistics.**  Lag-1 autocorrelation of daily returns
   and the average next-day return after an up day, a down day, two up
   days, two down days, and after three-day moves larger than 1.5
   standard deviations.  If short-window momentum existed, returns after
   up moves would be higher than after down moves.
2. **Short-window momentum rule.**  Hold the stock when its return over
   the last ``k`` days is positive, otherwise hold cash, for ``k`` from 1
   to 252.  The signal is read at the close and the trade is executed at
   the next open, with a one-way cost per switch.
3. **Short-window reversal rule.**  The mirror image: hold the stock when
   its return over the last ``k`` days is negative.
4. **Buy and hold** as the reference.

Findings (data to 13 November 2026)
-----------------------------------
* The semiconductor ETF (SMH) shows *negative* lag-1 autocorrelation
  (-0.10 since 2005).  Its average next-day return is 0 bps after an up
  day and +18 bps after a down day (t = 4.5), and +37 bps after a sharp
  three-day fall (t = 3.0).  At horizons of one to five days the sector
  mean-reverts.
* Momentum rules with ``k`` between 1 and 5 days lose to buy and hold in
  every multi-year sample.  Since 2005 they beat buy and hold in 0 or 1
  of 19 large semiconductor stocks, with a median Sharpe ratio between
  -0.05 and 0.18 against 0.57 for holding.
* In 2026 alone the 2-day and 3-day rules roughly matched buy and hold
  on SMH with a smaller drawdown.  The neighbouring 5-day rule lost
  money over the same months.  A result that flips sign when the
  parameter moves from 3 to 5 on 218 observations is noise.
* Trend rules start to hold up from about 3 months (63 days) upwards,
  where they give a Sharpe ratio similar to buy and hold with roughly
  half the drawdown.  That is what ``thematic_sleeve.py`` implements.

The impression that short-window momentum would have been very
profitable comes from hindsight: on a chart one sees the rallies that
continued and forgets the equally frequent strong days that were
followed by a reversal.  This script exists so that the claim can be
re-checked whenever the data changes.

Usage::

    python short_horizon_study.py

Outputs (written to ``data/research/``):
  - ``short_horizon_persistence_YYYY-MM-DD.csv``
  - ``short_horizon_rules_YYYY-MM-DD.csv``

References
----------
* Lehmann, B. (1990). Fads, martingales, and market efficiency.
  Quarterly Journal of Economics, 105(1), 1-28.
* Medhat, M., & Schmeling, M. (2022). Short-term momentum. Review of
  Financial Studies, 35(3), 1480-1526.
* Gao, C., Li, S. Z., Yuan, P., & Zhou, G. (2026). The anatomy of
  industry momentum. SSRN working paper 6371558.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from performance_metrics import compute_performance_metrics

STUDY_TICKERS: List[str] = ["SMH", "SOXX", "NVDA", "MU", "AVGO", "MRVL", "AMD", "TSM", "SPY", "QQQ"]
PERIOD_STARTS: List[str] = ["2005-01-01", "2016-01-01", "2023-01-01", "2026-01-01"]
HORIZONS: List[int] = [1, 2, 3, 5, 10, 21, 63, 126, 252]
RISK_FREE_TICKER = "^IRX"      # 13-week Treasury bill yield, in percent
MIN_HISTORY = 260


def download_open_close(tickers: Iterable[str], start: str = "1999-01-01") -> Dict[str, pd.DataFrame]:
    """Download split- and dividend-adjusted open and close prices.

    Parameters
    ----------
    tickers : iterable of str
        Symbols to download from Yahoo Finance.
    start : str, optional
        First date requested.

    Returns
    -------
    dict of pandas.DataFrame
        One DataFrame per ticker with columns ``Open`` and ``Close``, both
        adjusted, indexed by date.
    """
    import yfinance as yf

    out: Dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        data = yf.download(tickers=tkr, start=start, interval="1d", auto_adjust=True, progress=False)
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(tkr, axis=1, level=1, drop_level=True)
        out[tkr] = data[["Open", "Close"]].dropna()
    return out


def download_risk_free(start: str = "1999-01-01") -> pd.Series:
    """Daily risk-free return derived from the 13-week Treasury bill yield."""
    import yfinance as yf

    data = yf.download(tickers=RISK_FREE_TICKER, start=start, interval="1d", auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(RISK_FREE_TICKER, axis=1, level=1, drop_level=True)
    return (data["Close"] / 100.0 / 252.0).rename("risk_free")


# ---------------------------------------------------------------------------
# Persistence statistics
# ---------------------------------------------------------------------------

def _mean_bps_and_tstat(sample: pd.Series) -> tuple[float, int, float]:
    """Mean in basis points, sample size and t-statistic of the mean."""
    n = int(sample.count())
    if n < 6:
        return float("nan"), n, float("nan")
    mean = float(sample.mean())
    return 1e4 * mean, n, mean / (float(sample.std()) / np.sqrt(n))


def persistence_statistics(close: pd.Series, start: str) -> Dict[str, float]:
    """Autocorrelation and conditional next-day returns of one asset.

    Parameters
    ----------
    close : pandas.Series
        Adjusted close prices.
    start : str
        First date of the evaluation period.

    Returns
    -------
    dict
        ``autocorr_lag1`` plus, for each condition, the mean next-day
        return in basis points (``*_bps``), the number of observations
        (``*_n``) and the t-statistic (``*_t``).
    """
    r = close.pct_change()
    nxt = r.shift(-1)
    three_day = close / close.shift(3) - 1.0
    three_day_sd = r.rolling(21).std() * np.sqrt(3.0)
    conditions = {
        "all_days": pd.Series(True, index=r.index),
        "after_up_day": r > 0,
        "after_down_day": r <= 0,
        "after_two_up_days": (r > 0) & (r.shift(1) > 0),
        "after_two_down_days": (r < 0) & (r.shift(1) < 0),
        "after_sharp_3d_rise": three_day > 1.5 * three_day_sd,
        "after_sharp_3d_fall": three_day < -1.5 * three_day_sd,
    }
    window = (r.index >= pd.Timestamp(start)) & nxt.notna().to_numpy()
    out = {"autocorr_lag1": float(r[window].corr(nxt[window]))}
    for name, mask in conditions.items():
        bps, n, t = _mean_bps_and_tstat(nxt[window & mask.to_numpy()])
        out[f"{name}_bps"], out[f"{name}_n"], out[f"{name}_t"] = bps, n, t
    return out


# ---------------------------------------------------------------------------
# Trading rules
# ---------------------------------------------------------------------------

def momentum_position(close: pd.Series, k: int) -> pd.Series:
    """1.0 when the return over the last ``k`` days is positive, else 0.0."""
    return (close > close.shift(k)).astype(float)


def reversal_position(close: pd.Series, k: int) -> pd.Series:
    """1.0 when the return over the last ``k`` days is negative, else 0.0."""
    return (close < close.shift(k)).astype(float)


def run_next_open_rule(
    prices: pd.DataFrame,
    position: pd.Series,
    risk_free: Optional[pd.Series] = None,
    cost_bps: float = 5.0,
) -> pd.Series:
    """Daily returns of a long/flat rule executed at the next open.

    The position decided with the close of day ``t`` is established at
    the open of day ``t + 1``.  The return of day ``t + 1`` is therefore
    the overnight move earned by the *previous* position plus the
    open-to-close move earned by the *new* position.

    Parameters
    ----------
    prices : pandas.DataFrame
        Adjusted ``Open`` and ``Close`` of one asset.
    position : pandas.Series
        Target exposure in [0, 1] decided at each close.
    risk_free : pandas.Series, optional
        Daily risk-free return credited on the uninvested fraction.
    cost_bps : float, optional
        One-way cost in basis points per unit of exposure traded.

    Returns
    -------
    pandas.Series
        Net daily returns indexed by the day on which they are earned.
    """
    close, open_ = prices["Close"], prices["Open"]
    new_pos = position.shift(1).fillna(0.0)        # decided at t, applied on t + 1
    old_pos = position.shift(2).fillna(0.0)
    overnight = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    rf = risk_free.reindex(close.index).fillna(0.0) if risk_free is not None else 0.0
    ret = (1.0 + old_pos * overnight) * (1.0 + new_pos * intraday) - 1.0
    ret = ret - (new_pos - old_pos).abs() * cost_bps / 10_000.0 + (1.0 - new_pos) * rf
    return ret.iloc[1:]


def evaluate_rules(
    prices: pd.DataFrame,
    start: str,
    risk_free: Optional[pd.Series] = None,
    horizons: Iterable[int] = HORIZONS,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Compare momentum and reversal rules with buy and hold for one asset.

    Returns
    -------
    pandas.DataFrame
        One row per rule with Sharpe ratio, annualised return, maximum
        drawdown, average exposure and switches per year.
    """
    close = prices["Close"]
    first = max(MIN_HISTORY + 1, int(close.index.searchsorted(pd.Timestamp(start))) + 1)
    rf = risk_free.reindex(close.index).fillna(0.0) if risk_free is not None else None
    rows = []

    def add(name: str, ret: pd.Series, pos: Optional[pd.Series]) -> None:
        ret = ret.loc[close.index[first]:]
        m = compute_performance_metrics(ret, rf)
        held = pos.shift(1).loc[ret.index] if pos is not None else None
        rows.append({
            "rule": name,
            "sharpe_ratio": m["sharpe_ratio"],
            "annualised_return": m["annualised_return"],
            "max_drawdown": m["max_drawdown"],
            "avg_exposure": float(held.mean()) if held is not None else 1.0,
            "switches_per_year": float(held.diff().abs().sum() / (len(ret) / 252.0)) if held is not None else 0.0,
        })

    add("buy_and_hold", close.pct_change().iloc[1:], None)
    for k in horizons:
        pos = momentum_position(close, k)
        add(f"momentum_{k}d", run_next_open_rule(prices, pos, rf, cost_bps), pos)
    for k in [h for h in horizons if h <= 10]:
        pos = reversal_position(close, k)
        add(f"reversal_{k}d", run_next_open_rule(prices, pos, rf, cost_bps), pos)
    return pd.DataFrame(rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "data" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = download_open_close(STUDY_TICKERS)
    risk_free = download_risk_free()
    print(f"Downloaded {len(data)} tickers")

    persistence_rows, rule_frames = [], []
    for tkr, prices in data.items():
        for start in PERIOD_STARTS:
            stats = persistence_statistics(prices["Close"], start)
            persistence_rows.append({"ticker": tkr, "period_start": start, **stats})
            rules = evaluate_rules(prices, start, risk_free)
            rules.insert(0, "period_start", start)
            rules.insert(0, "ticker", tkr)
            rule_frames.append(rules)

    persistence = pd.DataFrame(persistence_rows)
    rules = pd.concat(rule_frames, ignore_index=True)
    today_str = dt.date.today().isoformat()
    persistence.to_csv(out_dir / f"short_horizon_persistence_{today_str}.csv", index=False)
    rules.to_csv(out_dir / f"short_horizon_rules_{today_str}.csv", index=False)

    print()
    print("Average next-day return (bps) after an up day versus a down day")
    cols = ["ticker", "period_start", "autocorr_lag1", "after_up_day_bps", "after_down_day_bps",
            "after_sharp_3d_rise_bps", "after_sharp_3d_fall_bps"]
    print(persistence[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print()
    print("Sharpe ratio by rule (rows) and period (columns), per ticker")
    for tkr in data:
        table = rules[rules["ticker"] == tkr].pivot(index="rule", columns="period_start", values="sharpe_ratio")
        order = ["buy_and_hold"] + [f"momentum_{k}d" for k in HORIZONS] + [f"reversal_{k}d" for k in HORIZONS if k <= 10]
        print(f"\n{tkr}")
        print(table.reindex(order).to_string(float_format=lambda v: f"{v:+.2f}"))

    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
