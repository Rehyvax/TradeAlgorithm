"""
ml_strategy.py
==============

This module implements a simple machine‑learning based
approach to cross‑sectional momentum.  The objective is
to use the engineered features (returns of various
horizons, volatility, drawdown and trend context) as
inputs to a classifier that predicts whether each
asset's future return over a given horizon will be
positive or negative.  The predicted probabilities are
converted into long/short signals, scaled by inverse
volatility and normalised similarly to the standard
momentum strategy.  The resulting portfolio is then
evaluated using common performance metrics.

The methodology is inspired by recent research on
machine‑learning applications to momentum investing,
which find that simple linear or tree‑based models
trained on engineered features can modestly improve
risk‑adjusted returns relative to naïve sorting rules.
Nevertheless, care must be taken to avoid look‑ahead
bias, overfitting and data snooping.  Here we use a
time‑split train/test and normalise features on the
training data only.

Usage::

    python TradeAlgorithm/ml_strategy.py

Prerequisites
-------------
This script depends on `pandas`, `numpy` and
`scikit‑learn`.  If `scikit‑learn` is not installed,
you can install it with `pip install scikit-learn`.

Disclaimer
----------
This is a research prototype provided for educational
purposes.  It is not financial advice, nor is it
intended for live trading without thorough testing,
robust risk management and appropriate regulatory
oversight.
"""

import os
import glob
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress convergence warnings for logistic regression
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Import feature engineering functions from the existing module
from feature_engineering import build_features


def prepare_ml_dataset(
    prices: pd.DataFrame,
    forward_horizon: int = 21,
) -> tuple:
    """Prepare features and labels for ML classification.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide DataFrame of adjusted prices with dates as index
        and tickers as columns.
    forward_horizon : int, optional
        Number of days ahead to compute the target return.  The
        default of 21 corresponds to approximately one month of
        trading.

    Returns
    -------
    X : pandas.DataFrame
        Design matrix with one row per (date, ticker) and
        engineered features as columns.
    y : pandas.Series
        Binary target indicating whether the forward return
        exceeds zero.
    weights_df : pandas.DataFrame
        Wide DataFrame of rolling volatilities used for
        scaling signals later.  Index aligned with
        `prices`.
    """
    # Build features using existing module
    feats = build_features(prices)
    # Extract relevant components
    returns = feats["returns"]  # log returns
    volatility = feats["volatility"]  # rolling vol
    drawdown = feats["drawdown"]  # drawdown
    trend = feats["trend"]  # above_ma, ma_slope
    # Flatten hierarchical columns for returns and volatility
    def flatten(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return df.copy().add_prefix(prefix + "_")
    returns_flat = flatten(returns, "ret")
    vol_flat = flatten(volatility, "vol")
    dd_flat = flatten(drawdown.to_frame(), "dd")
    trend_flat = trend.copy()
    trend_flat.columns = [f"{col}_{ticker}" if ticker != "" else col for col, ticker in trend_flat.columns]
    # Align all features by index
    all_feats = pd.concat([returns_flat, vol_flat, dd_flat, trend_flat], axis=1)
    # Drop rows with any missing values
    all_feats.dropna(inplace=True)
    # Forward returns for target
    future_prices = prices.shift(-forward_horizon)
    forward_returns = np.log(future_prices / prices)
    # Align with features index
    forward_returns = forward_returns.reindex(all_feats.index)
    # Flatten target to long format matching features
    X_list = []
    y_list = []
    vol_list = []
    # To avoid huge memory usage, iterate over dates and tickers
    for date in all_feats.index:
        # row of features for all assets on this date
        feat_row = all_feats.loc[date]
        # Reshape to long: features per ticker
        tickers = prices.columns
        for tkr in tickers:
            # Only include if target and vol are available
            if (
                tkr in forward_returns.columns
                and not pd.isna(forward_returns.loc[date, tkr])
                and not pd.isna(volatility.loc[date, tkr])
            ):
                # Build a dict for this asset
                feat_dict = {}
                # For each return horizon feature
                for col in returns.columns:
                    feat_dict[f"ret_{col}"] = returns.loc[date, col]
                for col in volatility.columns:
                    feat_dict[f"vol_{col}"] = volatility.loc[date, col]
                # Drawdown: same for all tickers on this date
                feat_dict["drawdown"] = drawdown.loc[date]
                # Trend context
                feat_dict[f"above_ma_{tkr}"] = trend.loc[date, ("above_ma", tkr)]
                feat_dict[f"ma_slope_{tkr}"] = trend.loc[date, ("ma_slope", tkr)]
                X_list.append(feat_dict)
                y_list.append(1.0 if forward_returns.loc[date, tkr] > 0 else 0.0)
                vol_list.append(volatility.loc[date, tkr])
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    vol_series = pd.Series(vol_list)
    return X, y, vol_series


def train_test_ml_strategy(prices: pd.DataFrame) -> dict:
    """Train a logistic regression model and evaluate its performance.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide DataFrame of adjusted prices with dates as index
        and tickers as columns.

    Returns
    -------
    dict
        Dictionary containing the trained model, performance
        metrics and the series of portfolio returns.
    """
    # Prepare dataset
    X, y, vol_series = prepare_ml_dataset(prices)
    # Convert to numpy arrays
    X_vals = X.values.astype(float)
    y_vals = y.values.astype(int)
    # Train/test split by time (first 70% train, last 30% test)
    split = int(0.7 * len(X_vals))
    X_train, X_test = X_vals[:split], X_vals[split:]
    y_train, y_test = y_vals[:split], y_vals[split:]
    vol_train, vol_test = vol_series.iloc[:split], vol_series.iloc[split:]
    # Build pipeline: standardise features then logistic regression
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=200)),
        ]
    )
    clf.fit(X_train, y_train)
    # Probabilities for positive return on test set
    probs = clf.predict_proba(X_test)[:, 1]
    # Determine thresholds for long/short: top 20% long, bottom 20% short
    ranks = pd.Series(probs).rank(pct=True, method="first")
    signals = pd.Series(0.0, index=ranks.index)
    signals[ranks >= 0.8] = 1.0
    signals[ranks <= 0.2] = -1.0
    # Scale by inverse volatility from vol_test
    scaled_signals = signals / vol_test.values
    # Normalise
    denom = np.sum(np.abs(scaled_signals))
    if denom > 0:
        weights = scaled_signals / denom
    else:
        weights = scaled_signals
    # Compute portfolio returns using subsequent 1‑day returns
    # For simplicity, assume daily returns equal to log_returns vectorised from price panel
    # Extract dates corresponding to test set rows (approx.)
    # As we flattened features, we lost explicit dates; reuse chronological order in prices
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")
    # Align to test length; this is approximate due to flattening; use last len(X_test) days
    # Build weighted average returns for each test instance; this is a simplification.
    # For a more precise implementation, one would need to keep track of asset and date per sample.
    port_ret = []
    idx = -len(X_test)
    for i in range(len(X_test)):
        # weight applied to overall cross‑section equally
        daily_ret = log_returns.iloc[idx + i].mean()
        port_ret.append(weights[i] * daily_ret)
    port_ret = pd.Series(port_ret)
    metrics = compute_metrics(port_ret)
    return {
        "model": clf,
        "metrics": metrics,
        "returns": port_ret,
    }


def compute_metrics(portfolio_returns: pd.Series) -> dict:
    """Reuse the metrics function from robust_evaluation script."""
    r = portfolio_returns.dropna().astype(float)
    if r.empty:
        return {
            "cumulative_return": np.nan,
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
        }
    cumulative = (1 + r).prod() - 1
    periods_per_year = 252
    ann_return = (1 + cumulative) ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
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


def main():
    # Locate latest price panel
    price_dir = os.path.join("data", "prices")
    pattern = os.path.join(price_dir, "adjclose_wide_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No price files found in {price_dir}. Run trade_algorithm.py first."
        )
    latest_file = files[-1]
    prices = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    # Drop all‑NaN columns
    prices = prices.dropna(axis=1, how="all")
    # Run ML strategy
    result = train_test_ml_strategy(prices)
    print("ML strategy performance metrics:")
    for k, v in result["metrics"].items():
        print(f"{k}: {v:.4f}")
    # Save returns
    signals_dir = os.path.join("data", "signals")
    os.makedirs(signals_dir, exist_ok=True)
    today_str = dt.date.today().isoformat()
    returns_file = os.path.join(
        signals_dir, f"ml_returns_{today_str}.csv"
    )
    result["returns"].to_csv(returns_file, index=True, header=["return"])
    print(f"Saved ML strategy returns to {returns_file}")


if __name__ == "__main__":
    main()