"""
thematic_sleeve.py
==================

Risk-managed exposure to a user-defined theme (by default AI and
semiconductor stocks listed in ``data/universe_thematic.csv``).

Purpose
-------
The core strategy (``trend_strategy.py``) trades broad asset-class ETFs.
This sleeve covers a different need: holding a concentrated, volatile
theme without sitting through its full drawdowns.  The choice of theme is
a discretionary view and the sleeve does not claim to add return to it.
What it does is decide *how much* of the theme to hold each week.

Rules
-----
For every stock in the theme:

1. **Volatility scaling.**  Base exposure is
   ``min(1, asset_target_vol / realised_vol)`` using 21-day volatility.
   A stock running at 60 % annualised volatility gets half the capital of
   one running at 30 %.
2. **Trend gate with a floor.**  The base exposure is multiplied by
   ``floor + (1 - floor) * score`` where ``score`` is the fraction of the
   3, 6 and 12 month returns that are positive.  A stock in a clear
   downtrend keeps only the floor (25 % by default) instead of being sold
   completely, because V-shaped recoveries are frequent in this sector
   and a full exit systematically misses them.
3. **Equal capital per name** before the two scalings, weekly rebalance.

An optional five-day reversal tilt is available (``reversal_tilt``).  It
is off by default; ``short_horizon_study.py`` documents why.

Evidence
--------
Tested on daily data to 13 November 2026 (see README, section "Thematic
sleeve"), the sleeve delivered roughly the same Sharpe ratio as holding
the equal-weighted basket, with about half the volatility and a maximum
drawdown near -25 % instead of -67 %.  It is a risk-control overlay, not
a source of alpha.

The 19 default tickers are today's large semiconductor companies.  A
backtest on today's survivors overstates the return of the *basket*; the
comparison between the sleeve and holding the same basket is not
affected by that bias in the same way, and that comparison is the claim
made here.

Usage::

    python thematic_sleeve.py

Outputs (written to ``data/signals/``):
  - ``weights_thematic_YYYY-MM-DD.csv``   daily target weights
  - ``thematic_returns_YYYY-MM-DD.csv``   daily net returns of sleeve and basket
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from performance_metrics import compute_performance_metrics


@dataclass(frozen=True)
class SleeveConfig:
    """Configuration of the thematic sleeve.

    Parameters
    ----------
    asset_target_vol : float
        Annualised volatility at which a single stock receives its full
        equal-weight allocation.
    vol_window : int
        Window, in trading days, for realised volatility.
    lookbacks : tuple of int
        Lookbacks of the trend score.
    floor : float
        Fraction of the base exposure kept when every lookback is
        negative.
    reversal_tilt : float
        Strength of the five-day reversal tilt.  Zero disables it.
    rebalance_every : int
        Trading days between rebalances.
    min_history : int
        Rows of history required before a stock is eligible.
    cost_bps : float
        One-way cost in basis points of traded notional.
    """

    asset_target_vol: float = 0.30
    vol_window: int = 21
    lookbacks: Tuple[int, ...] = (63, 126, 252)
    floor: float = 0.25
    reversal_tilt: float = 0.0
    rebalance_every: int = 5
    min_history: int = 252
    cost_bps: float = 5.0


def compute_asset_exposure(prices: pd.DataFrame, cfg: SleeveConfig = SleeveConfig()) -> pd.DataFrame:
    """Exposure of each stock, between 0 and 1, before dividing capital.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices (dates x tickers).
    cfg : SleeveConfig, optional
        Sleeve configuration.

    Returns
    -------
    pandas.DataFrame
        Exposures in [0, 1]; NaN where a stock lacks ``min_history`` rows.
    """
    returns = prices.pct_change()
    # Root mean square of daily returns (zero-mean volatility estimate).
    vol = np.sqrt((returns ** 2).rolling(cfg.vol_window).mean() * 252.0)
    exposure = (cfg.asset_target_vol / vol).clip(upper=1.0)
    score = sum((prices > prices.shift(lb)).astype(float) for lb in cfg.lookbacks) / len(cfg.lookbacks)
    exposure = exposure * (cfg.floor + (1.0 - cfg.floor) * score)
    if cfg.reversal_tilt:
        z = (prices / prices.shift(5) - 1.0) / (vol / np.sqrt(252.0) * np.sqrt(5.0))
        exposure = exposure * (1.0 - cfg.reversal_tilt * z.clip(-1.0, 1.0))
    eligible = prices.shift(cfg.min_history).notna()
    return exposure.clip(lower=0.0, upper=1.0).where(eligible)


def build_sleeve_weights(
    prices: pd.DataFrame,
    cfg: SleeveConfig = SleeveConfig(),
    start: Optional[str] = None,
) -> pd.DataFrame:
    """Daily sleeve weights, held constant between rebalance dates.

    Each eligible stock receives ``exposure / n_eligible``.  The weights
    therefore sum to less than one; the remainder is cash.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices.
    cfg : SleeveConfig, optional
        Sleeve configuration.
    start : str, optional
        Date from which the rebalance schedule is counted.  Defaults to
        the first row with at least one eligible stock.
    """
    exposure = compute_asset_exposure(prices, cfg)
    n_eligible = exposure.notna().sum(axis=1)
    target = exposure.div(n_eligible.replace(0, np.nan), axis=0).fillna(0.0)
    first = prices.index.searchsorted(pd.Timestamp(start)) if start else int(np.argmax(n_eligible.to_numpy() > 0))
    steps = np.arange(len(prices)) - first
    is_rebalance = (steps >= 0) & (steps % cfg.rebalance_every == 0)
    weights = target.where(pd.Series(is_rebalance, index=prices.index), np.nan)
    weights.iloc[:first] = 0.0
    return weights.ffill().fillna(0.0)


def latest_sleeve_weights(prices: pd.DataFrame, cfg: SleeveConfig = SleeveConfig()) -> pd.Series:
    """Sleeve weights as of the last row of ``prices`` (for live execution)."""
    exposure = compute_asset_exposure(prices, cfg).iloc[-1]
    n = int(exposure.notna().sum())
    return (exposure / n).fillna(0.0) if n else exposure.fillna(0.0)


def run_thematic_sleeve(
    prices: pd.DataFrame,
    cfg: SleeveConfig = SleeveConfig(),
    risk_free: Optional[pd.Series] = None,
    start: Optional[str] = None,
) -> dict:
    """Backtest the sleeve against holding the equal-weighted basket.

    Weights decided with the close of day ``t`` become effective at the
    close of day ``t + 1``.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices of the theme.
    cfg : SleeveConfig, optional
        Sleeve configuration.
    risk_free : pandas.Series, optional
        Daily risk-free return, credited on the uninvested fraction and
        subtracted in the Sharpe ratio.
    start : str, optional
        First date of the evaluation.

    Returns
    -------
    dict
        ``metrics``, ``basket_metrics``, ``weights``, ``returns`` and
        ``basket_returns``.
    """
    prices = prices.sort_index().ffill()
    asset_returns = prices.pct_change().fillna(0.0)
    rf = risk_free.reindex(prices.index).fillna(0.0) if risk_free is not None else pd.Series(0.0, index=prices.index)
    weights = build_sleeve_weights(prices, cfg, start)
    held = weights.shift(1).fillna(0.0)            # one-day implementation lag
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    invested = held.shift(1).fillna(0.0)
    net = (invested * asset_returns).sum(axis=1) + (1.0 - invested.sum(axis=1)) * rf
    net = net - turnover * cfg.cost_bps / 10_000.0
    eligible = prices.shift(cfg.min_history).notna().shift(1, fill_value=False)
    basket = asset_returns.where(eligible).mean(axis=1).fillna(0.0)
    if start:
        net, basket, rf_eval = net.loc[start:].iloc[1:], basket.loc[start:].iloc[1:], rf.loc[start:].iloc[1:]
    else:
        rf_eval = rf
    return {
        "metrics": compute_performance_metrics(net, rf_eval),
        "basket_metrics": compute_performance_metrics(basket, rf_eval),
        "weights": weights,
        "returns": net,
        "basket_returns": basket,
    }


def load_thematic_universe(base_dir: Path) -> list[str]:
    """Read the tickers of ``data/universe_thematic.csv``."""
    universe = pd.read_csv(base_dir / "data" / "universe_thematic.csv")
    return universe["Ticker"].dropna().astype(str).str.strip().unique().tolist()


def download_thematic_prices(base_dir: Path, start: str = "2003-01-01") -> pd.DataFrame:
    """Download adjusted closes for the theme and save them under ``data/prices``.

    The file is named ``thematic_adjclose_wide_YYYY-MM-DD.csv`` so that it
    never collides with the ETF panel used by the core strategy.
    """
    from loaders.price_loader import PriceDownloadConfig, download_prices_adjclose

    tickers = load_thematic_universe(base_dir)
    long = download_prices_adjclose(tickers, PriceDownloadConfig(start=start))
    if long.empty:
        raise ValueError("No thematic prices were downloaded.")
    wide = long.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
    out_dir = base_dir / "data" / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / f"thematic_adjclose_wide_{dt.date.today().isoformat()}.csv")
    return wide


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    prices = download_thematic_prices(base_dir)
    print(f"Loaded: {len(prices)} rows x {len(prices.columns)} tickers")
    print(f"Range : {prices.index[0].date()} -- {prices.index[-1].date()}")

    # Daily risk-free return from the ETF panel's cash ticker when available.
    risk_free = None
    etf_files = sorted((base_dir / "data" / "prices").glob("adjclose_wide_*.csv"))
    if etf_files:
        etf = pd.read_csv(etf_files[-1], index_col=0, parse_dates=True)
        if "SHY" in etf.columns:
            risk_free = etf["SHY"].pct_change()

    print()
    print(f"{'period':<12} {'':<8} {'Sharpe':>7} {'return':>8} {'vol':>7} {'max DD':>8}")
    for start in ["2006-01-03", "2016-01-04", "2023-01-03"]:
        res = run_thematic_sleeve(prices, risk_free=risk_free, start=start)
        for label, m in [("sleeve", res["metrics"]), ("basket", res["basket_metrics"])]:
            print(f"{start:<12} {label:<8} {m['sharpe_ratio']:>7.2f} {m['annualised_return']:>8.1%} "
                  f"{m['annualised_volatility']:>7.1%} {m['max_drawdown']:>8.1%}")

    res = run_thematic_sleeve(prices, risk_free=risk_free, start="2006-01-03")
    signals_dir = base_dir / "data" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    today_str = dt.date.today().isoformat()
    res["weights"].to_csv(signals_dir / f"weights_thematic_{today_str}.csv")
    pd.DataFrame({"sleeve": res["returns"], "basket": res["basket_returns"]}).to_csv(
        signals_dir / f"thematic_returns_{today_str}.csv"
    )
    print(f"\nSaved weights and returns to {signals_dir}")
    print("\nCurrent target weights (fraction of the sleeve):")
    print(latest_sleeve_weights(prices).sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
