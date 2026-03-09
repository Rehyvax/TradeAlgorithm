"""
Feature engineering utilities for financial time‑series data.

This module contains a collection of functions to transform raw price
data into a set of engineered features that are commonly used in
quantitative investing and research.  The goal of feature engineering
is to extract information from historical prices that can be used to
drive trading signals or risk models.  For example, time‑series and
cross‑sectional momentum strategies rely on past returns, moving
averages and volatility measures to gauge the strength and persistence
of trends【573758359642248†L34-L43】, while risk management routines use
drawdowns and volatility scaling to adjust position sizes【108184354453101†L89-L100】.

The functions in this file operate on pandas DataFrames where the
rows are dates (or datetimes) and the columns are asset identifiers
(e.g. tickers).  All computations are vectorised for efficiency and
use simple rolling statistics that do not introduce look‑ahead bias.
They are meant to be building blocks for more complex models and
portfolio construction frameworks, as described in the TrendFolios®
paper where trend‐following strategies analyse historical prices and
returns across multiple timeframes to produce indicators based on
moving averages, volatility measures and price ranges【573758359642248†L34-L43】.

Functions
---------
compute_log_returns : Compute log returns over multiple horizons.
compute_rolling_volatility : Compute rolling standard deviation of
    daily log returns.
compute_rolling_drawdown : Compute rolling drawdown from peak.
compute_trend_features : Compute moving‑average based trend signals.
compute_cross_sectional_rank : Compute cross‑sectional percentile rank across assets.
build_features : Combine all of the above into a single DataFrame.

Usage example
-------------
Assuming you have a DataFrame ``prices`` where each column is a
different asset’s adjusted close price and the index is a DateTime
Index, you can generate a feature matrix as follows::

    from feature_engineering import build_features
    features = build_features(prices,
                              return_horizons=[1, 5, 21, 63, 252],
                              vol_windows=[21, 63, 252],
                              ma_window=200,
                              drawdown_window=252)

The result is a dictionary of DataFrames keyed by feature name.  You
can join them into a single multi‑index DataFrame or keep them
separate depending on your workflow.

References
----------
* "TrendFolios®: A Portfolio Construction Framework for Utilizing
  Momentum and Trend‑Following In a Multi‑Asset Portfolio" (Lu et al.,
  2025) — Discusses the importance of moving averages, volatility
  measures and trading ranges in trend‑following strategies【573758359642248†L34-L43】.
* "Feature Engineering for Financial Market Prediction: From Historical
  Data to Actionable Insights" (Ailyn, 2024) — Explains that features
  can include price trends, trading volumes and other time‑series
  specific variables such as moving averages and volatility measures【949929375550076†L35-L50】.
* "Financial Engineering and Time Series Analytics" (Jones, 2025) —
  Notes that key feature engineering techniques for financial time
  series include lag features, differencing, volatility measures and
  momentum indicators【260462117398459†L48-L52】.
* "Enhancing Time Series Momentum Strategies Using Deep Neural
  Networks" (Lim et al., 2019) — Highlights the role of volatility
  scaling in improving the performance of momentum strategies【108184354453101†L89-L100】.

"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def compute_log_returns(
    prices: pd.DataFrame, horizons: Iterable[int] = (1, 5, 21, 63, 252)
) -> Dict[str, pd.DataFrame]:
    """Compute log returns over multiple horizons.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide DataFrame of prices with dates as the index and tickers as
        columns.  Prices should be positive and correspond to the same
        time grid (e.g. daily adjusted close).
    horizons : iterable of int, optional
        Number of days over which to compute log returns.  Typical
        values include 1 (daily), 5 (weekly), 21 (monthly), 63 (3 months)
        and 252 (one year).  Larger horizons capture longer term
        momentum effects【573758359642248†L34-L43】.

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary where each key is ``"ret_{horizon}d"`` and each value
        is a DataFrame of log returns for that horizon.  The index and
        columns match ``prices``.  Missing values arise at the start of
        the series due to the shift.
    """
    returns: Dict[str, pd.DataFrame] = {}
    # Use natural logarithms to compute returns; log returns are
    # additive over time and less sensitive to extreme values than
    # simple percentage returns.  This is common in finance research.
    for h in horizons:
        # Shift prices by h periods and compute log ratio.
        shifted = prices.shift(h)
        ret = np.log(prices / shifted)
        returns[f"ret_{h}d"] = ret
    return returns


def compute_rolling_volatility(
    log_returns: pd.DataFrame, windows: Iterable[int] = (21, 63, 252)
) -> Dict[str, pd.DataFrame]:
    """Compute rolling volatility of daily log returns.

    Volatility is measured as the rolling standard deviation of daily
    log returns.  Different windows capture different risk horizons.
    Empirical research shows that incorporating volatility measures
    alongside momentum signals can improve risk‑adjusted performance
    【108184354453101†L89-L100】.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        DataFrame of daily log returns.  Typically the output of
        ``compute_log_returns`` with horizon=1.
    windows : iterable of int, optional
        Rolling window sizes (in days) over which to compute the
        standard deviation.  Common choices are 21 (1 month), 63
        (3 months) and 252 (1 year).

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary where each key is ``"vol_{window}d"`` and each value
        is a DataFrame of rolling standard deviations.
    """
    vols: Dict[str, pd.DataFrame] = {}
    for w in windows:
        vol = log_returns.rolling(w).std()
        vols[f"vol_{w}d"] = vol
    return vols


def compute_rolling_drawdown(
    prices: pd.DataFrame, window: int = 252
) -> pd.DataFrame:
    """Compute rolling drawdown from the peak.

    Drawdown is defined as the percentage decline from the rolling
    maximum price over a given window.  It measures how far the
    current price has fallen relative to its recent highs and is
    commonly used for risk management and regime classification【573758359642248†L34-L43】.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide DataFrame of prices with dates as the index and tickers as
        columns.
    window : int, optional
        Number of days used to compute the rolling maximum.  A
        typical choice is 252 (approximately one year) to gauge
        drawdowns relative to the past year.

    Returns
    -------
    pandas.DataFrame
        DataFrame of drawdowns where values lie between -1 and 0.  A
        value of 0 indicates the price is at its rolling maximum,
        while -0.20 means the price is 20% below its 252‑day high.
    """
    rolling_max = prices.rolling(window).max()
    drawdown = prices / rolling_max - 1.0
    drawdown = drawdown.clip(lower=-1.0, upper=0.0)
    return drawdown


def compute_trend_features(
    prices: pd.DataFrame, ma_window: int = 200
) -> pd.DataFrame:
    """Compute moving‑average based trend features.

    Trend indicators compare the current price to a long‑term moving
    average to determine whether an asset is in an uptrend or downtrend.
    They also capture the slope of the moving average to measure the
    strength of the trend.  Such moving‑average rules are widely used
    in momentum and trend‑following strategies【573758359642248†L34-L43】.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide DataFrame of prices with dates as the index and tickers as
        columns.
    ma_window : int, optional
        Window length (in days) for the simple moving average.  A
        common choice is 200 days (roughly 9–10 months).

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns per asset: ``above_ma`` indicates
        whether the price is above the moving average (1.0 for True,
        0.0 for False) and ``ma_slope`` is the difference of the
        moving average from one period to the next.  The DataFrame
        has a MultiIndex on columns with level 0 being the feature
        name and level 1 being the asset.
    """
    # Compute the moving average
    ma = prices.rolling(ma_window).mean()
    # Boolean indicator: 1 if price above MA, else 0
    above_ma = (prices > ma).astype(float)
    # Slope of the moving average (first difference)
    ma_slope = ma.diff()
    # Combine into a MultiIndex DataFrame
    features = {}
    for col in prices.columns:
        features[("above_ma", col)] = above_ma[col]
        features[("ma_slope", col)] = ma_slope[col]
    combined = pd.DataFrame(features)
    combined.columns = pd.MultiIndex.from_tuples(combined.columns, names=["feature", "asset"])
    return combined


def compute_cross_sectional_rank(data: pd.DataFrame) -> pd.DataFrame:
    """Compute cross‑sectional percentile ranks across assets.

    Given a wide DataFrame where each row corresponds to a date and each
    column corresponds to an asset (e.g. a log return series), this
    function replaces each value by its percentile rank within the row.

    Cross‑sectional ranking is a key ingredient in relative momentum
    strategies: assets with the highest historical returns are ranked
    near 1.0 and those with the lowest returns are ranked near 0.0.
    Empirical studies show that constructing long‑short portfolios by
    buying assets with the highest‑ranked trend signals and selling
    those with the lowest‑ranked signals can deliver positive returns
    【416002924589760†L24-L33】.  Using percentile ranks rather than raw
    returns also makes the feature scale‑free and robust to outliers.

    Parameters
    ----------
    data : pandas.DataFrame
        Wide DataFrame of numbers indexed by dates and with columns
        representing assets.  Missing values are ignored in the
        ranking.  Rows with fewer than two non‑missing values return
        ``NaN`` ranks.

    Returns
    -------
    pandas.DataFrame
        DataFrame of the same shape as ``data`` where each entry is
        replaced by its percentile rank in the row, taking values in
        the interval (0, 1].
    """
    # pct=True returns percentile rank normalized to [0,1], ties are averaged
    return data.rank(axis=1, pct=True)


def build_features(
    prices: pd.DataFrame,
    return_horizons: Iterable[int] = (1, 5, 21, 63, 252),
    vol_windows: Iterable[int] = (21, 63, 252),
    ma_window: int = 200,
    drawdown_window: int = 252,
    cs_rank_horizons: Iterable[int] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Compute a suite of engineered features from price data.

    This convenience function orchestrates the calculation of log
    returns, rolling volatility, drawdown and trend indicators.  It
    returns a dictionary of DataFrames keyed by feature name so that
    downstream code can select or combine the features as needed.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide DataFrame of prices with dates as the index and tickers as
        columns.
    return_horizons : iterable of int, optional
        Horizons (in days) over which to compute log returns.  See
        ``compute_log_returns``.
    vol_windows : iterable of int, optional
        Rolling window sizes for volatility.  See
        ``compute_rolling_volatility``.
    ma_window : int, optional
        Window size for the moving average in trend features.
    drawdown_window : int, optional
        Window size for drawdown calculation.

    ``cs_rank_horizons`` specifies which return horizons should be
    converted into cross‑sectional percentile ranks.  For example,
    specifying ``(252,)`` will compute percentile ranks for the 252‑day
    (one year) log return across all assets on each date.  Cross‑
    sectional ranking is the basis of relative momentum strategies that
    buy past winners and sell past losers【416002924589760†L24-L33】.  If
    ``cs_rank_horizons`` is ``None``, no cross‑sectional ranks are
    computed.

    Returns
    -------
    dict of pandas.DataFrame
        Dictionary mapping feature group names to DataFrames.  Keys
        include ``'returns'``, ``'volatility'``, ``'drawdown'``, ``'trend'`` and
        optionally ``'cross_sectional_rank'`` if ranks were requested.  Each
        DataFrame has the same index as ``prices``.
    """
    # Ensure input is sorted by date to avoid look‑ahead bias
    prices = prices.sort_index()
    # Compute log returns for all horizons
    returns_dict = compute_log_returns(prices, horizons=return_horizons)
    # Daily returns for volatility
    daily_ret = returns_dict[f"ret_{return_horizons[0]}d"]
    # Compute rolling volatility of daily returns
    vol_dict = compute_rolling_volatility(daily_ret, windows=vol_windows)
    # Compute drawdown
    drawdown = compute_rolling_drawdown(prices, window=drawdown_window)
    # Compute trend features
    trend = compute_trend_features(prices, ma_window=ma_window)
    # Initialise the feature dictionary
    features: Dict[str, pd.DataFrame] = {}
    features["returns"] = pd.concat(returns_dict.values(), axis=1, keys=returns_dict.keys())
    features["volatility"] = pd.concat(vol_dict.values(), axis=1, keys=vol_dict.keys())
    features["drawdown"] = drawdown
    features["trend"] = trend
    # Optionally compute cross‑sectional ranks on selected horizons
    if cs_rank_horizons:
        cs_dict: Dict[str, pd.DataFrame] = {}
        for h in cs_rank_horizons:
            key = f"ret_{h}d"
            if key not in returns_dict:
                raise KeyError(f"Horizon {h} not in computed returns. Available: {list(returns_dict.keys())}")
            cs_dict[f"cs_rank_{h}d"] = compute_cross_sectional_rank(returns_dict[key])
        # Concatenate along a new column level keyed by horizon
        features["cross_sectional_rank"] = pd.concat(cs_dict.values(), axis=1, keys=cs_dict.keys())
    return features


if __name__ == "__main__":  # pragma: no cover
    # Example usage for manual testing
    import matplotlib.pyplot as plt

    # Generate a simple random price series for demonstration
    rng = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(0)
    # Simulate 3 assets with geometric Brownian motion
    prices = pd.DataFrame(
        {
            "AssetA": 100 * np.exp(np.cumsum(0.001 + 0.02 * np.random.randn(len(rng))), dtype=float),
            "AssetB": 50 * np.exp(np.cumsum(0.0015 + 0.025 * np.random.randn(len(rng))), dtype=float),
            "AssetC": 80 * np.exp(np.cumsum(0.0005 + 0.015 * np.random.randn(len(rng))), dtype=float),
        },
        index=rng,
    )
    feats = build_features(prices)
    # Plot one of the features as a quick sanity check
    feats["volatility"]["vol_21d"].plot(title="21‑day rolling volatility")
    plt.show()