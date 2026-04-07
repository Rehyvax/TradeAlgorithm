"""
robust_evaluation.py
====================

This module performs a grid search over several
parameter combinations for the cross‑sectional
momentum strategy previously implemented.  It
illustrates how a professional researcher might
stress‑test a trading rule across different look‑back
periods, ranking thresholds and volatility windows.

The script loads the most recent wide price panel
(`adjclose_wide_YYYY-MM-DD.csv`) from the local
`data/prices` directory, computes returns and
generates signals for each parameter set, then
evaluates performance metrics such as cumulative
return, annualised return, annualised volatility,
Sharpe ratio and maximum drawdown.  Results are
saved to `data/signals/robust_results_YYYY-MM-DD.csv`
for subsequent analysis.

The methodology follows best practices described in
the academic literature on cross‑sectional momentum
and risk management.  In particular, we use
multi‑horizon ranking of log returns to identify
relative winners and losers【416002924589760†L24-L33】
and scale positions inversely to recent volatility
to mitigate crash risk【416002924589760†L81-L113】.

Usage::

    python TradeAlgorithm/robust_evaluation.py

The script will search for price panels in
`data/prices`, run the evaluation and write
out a CSV with the metrics of each tested
combination.  It will also save the best
parameter combination according to Sharpe ratio.

Notes
-----
This script is meant for offline research.  It does
not place trades or interact with any broker.  For
live or paper trading execution, see the guidance in
the main assistant response.
"""

from __future__ import annotations

import os
import glob
import datetime as dt
from itertools import product

import numpy as np
import pandas as pd


def compute_metrics(portfolio_returns: pd.Series) -> dict:
    """Compute basic performance statistics for a series of
    periodic returns.

    Parameters
    ----------
    portfolio_returns : pandas.Series
        Series of daily (or periodic) portfolio returns.

    Returns
    -------
    dict
        Dictionary with cumulative return, annualised return,
        annualised volatility, Sharpe ratio (assumes zero
        risk‑free rate) and maximum drawdown.
    """
    # drop missing values
    r = portfolio_returns.dropna().astype(float)
    if r.empty:
        return {
            "cumulative_return": np.nan,
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
        }
    # cumulative return (end value / start value - 1)
    cumulative = (1 + r).prod() - 1
    # convert daily return to annualised return and vol
    periods_per_year = 252  # assuming trading days
    ann_return = (1 + cumulative) ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    # max drawdown
    cum_curve = (1 + r).cumprod()
    running_max = cum_curve.cummax()
    drawdown = cum_curve / running_max - 1.0
    max_dd = drawdown.min()
    return {
        "cumulative_return": cumulative,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def run_strategy(
    prices: pd.DataFrame,
    ranking_horizon: int,
    top_pct: float,
    bottom_pct: float,
    vol_window: int,
    volume_wide: pd.DataFrame | None = None,
) -> dict:
    """Execute a cross‑sectional momentum strategy for a given parameter set.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide DataFrame of adjusted prices with dates as the index
        (datetime) and tickers as columns.
    ranking_horizon : int
        Look‑back window (in trading days) used to compute the
        momentum signal.  Typical values: 63 (3m), 126 (6m),
        252 (12m).
    top_pct : float
        Percentile threshold for the long side.  Assets with
        momentum rank ≥ (1‑top_pct) are assigned +1 before
        scaling.
    bottom_pct : float
        Percentile threshold for the short side.  Assets with
        momentum rank ≤ bottom_pct are assigned –1 before
        scaling.
    vol_window : int
        Rolling window (in trading days) used to estimate
        historical volatility for scaling.  Typical values: 21,
        63, 126.
    volume_wide : pandas.DataFrame or None, optional
        Wide DataFrame of daily share volume aligned with ``prices``.
        When provided, transaction costs are computed with a
        square-root market-impact model::

            participation_rate = |Δweight| / (price × daily_volume)
            market_impact_bps  = 10 × √participation_rate
            daily_cost         = Σ market_impact_bps / 10 000

        When ``None``, a flat 5 bps charge on daily turnover is used.

    Returns
    -------
    dict
        Dictionary with metrics and additional outputs (signals,
        weights, returns).  See `compute_metrics` for metric
        definitions.
    """
    # Compute daily log returns
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")
    # Compute momentum over ranking_horizon
    momentum = np.log(prices / prices.shift(ranking_horizon))
    momentum = momentum.reindex(log_returns.index)
    # cross‑sectional rank (percentile) for each date
    rank = momentum.rank(axis=1, pct=True, method="average")
    # Signals: +1 for top percentile, –1 for bottom percentile, 0 elsewhere
    signals = pd.DataFrame(0.0, index=rank.index, columns=rank.columns)
    signals[rank >= (1.0 - top_pct)] = 1.0
    signals[rank <= bottom_pct] = -1.0
    # historical volatility for scaling
    vol = log_returns.rolling(vol_window).std()
    # avoid division by zero
    vol = vol.replace(0.0, np.nan)
    # scale signals by inverse volatility
    scaled_signals = signals.divide(vol, axis=0).fillna(0.0)
    # normalise so that sum of absolute weights = 1
    weight_norm = scaled_signals.abs().sum(axis=1).replace(0.0, np.nan)
    weights = scaled_signals.div(weight_norm, axis=0).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    gross = (weights.shift(1) * log_returns).sum(axis=1)

    if volume_wide is not None:
        # Square-root market-impact model
        p = prices.reindex(index=log_returns.index).reindex(columns=weights.columns)
        v = volume_wide.reindex(index=log_returns.index).reindex(columns=weights.columns)
        delta_w = weights.diff().abs()
        dollar_vol = p * v
        with np.errstate(divide="ignore", invalid="ignore"):
            part_rate = delta_w / dollar_vol
        part_rate = part_rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        impact_bps = 10.0 * np.sqrt(part_rate)
        cost = (impact_bps / 10_000.0).sum(axis=1)
        recorded_cost_bps = float(np.nan)
    else:
        # Flat 5 bps on turnover (fallback)
        recorded_cost_bps = 5.0
        cost = turnover * (recorded_cost_bps / 10_000.0)

    port_returns = gross - cost
    # portfolio returns: weights shifted by one day to avoid look‑ahead bias
    metrics = compute_metrics(port_returns)
    metrics["cost_bps"] = recorded_cost_bps
    metrics["avg_turnover"] = float(turnover.mean())
    metrics["total_cost"] = float(cost.sum())
    return {
        "metrics": metrics,
        "signals": signals,
        "weights": weights,
        "returns": port_returns,
    }


def main():
    # Locate the most recent adjclose_wide file
    price_dir = os.path.join("data", "prices")
    pattern = os.path.join(price_dir, "adjclose_wide_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No price files found in {price_dir}. Run trade_algorithm.py first."
        )
    latest_file = files[-1]
    prices = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    # drop any columns with all NaNs
    prices = prices.dropna(axis=1, how="all")
    # Load volume data for market-impact costs if available
    vol_files = sorted(glob.glob(os.path.join(price_dir, "volume_wide_*.csv")))
    volume_wide = None
    if vol_files:
        volume_wide = pd.read_csv(vol_files[-1], index_col=0, parse_dates=True)
    # Parameter grid
    ranking_options = [63, 126, 252]
    percentile_options = [0.1, 0.2, 0.3]
    vol_windows = [21, 63, 126]
    results = []
    # iterate over parameter combinations
    for rh, pct, vw in product(ranking_options, percentile_options, vol_windows):
        res = run_strategy(prices, rh, pct, pct, vw, volume_wide=volume_wide)
        metrics = res["metrics"].copy()
        metrics.update(
            {
                "ranking_horizon": rh,
                "top_pct": pct,
                "bottom_pct": pct,
                "vol_window": vw,
            }
        )
        results.append(metrics)
    results_df = pd.DataFrame(results)
    # Sort by Sharpe ratio descending
    results_df.sort_values(by="sharpe_ratio", ascending=False, inplace=True)
    # Save results
    signals_dir = os.path.join("data", "signals")
    os.makedirs(signals_dir, exist_ok=True)
    today_str = dt.date.today().isoformat()
    result_file = os.path.join(signals_dir, f"robust_results_{today_str}.csv")
    results_df.to_csv(result_file, index=False)
    print(f"Saved robust evaluation results to {result_file}")
    # Save best configuration weights and returns for inspection
    best = results_df.iloc[0]
    print("Best configuration:")
    print(best)
    best_run = run_strategy(
        prices,
        ranking_horizon=int(best["ranking_horizon"]),
        top_pct=float(best["top_pct"]),
        bottom_pct=float(best["bottom_pct"]),
        vol_window=int(best["vol_window"]),
        volume_wide=volume_wide,
    )
    # Save weights and returns for the best config
    best_weights_file = os.path.join(
        signals_dir, f"best_weights_{today_str}.csv"
    )
    best_run["weights"].to_csv(best_weights_file, index=True)
    best_returns_file = os.path.join(
        signals_dir, f"best_returns_{today_str}.csv"
    )
    best_run["returns"].to_csv(best_returns_file, index=True, header=["return"])
    print(
        f"Saved best weights to {best_weights_file} and returns to {best_returns_file}."
    )


if __name__ == "__main__":
    main()