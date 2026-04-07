"""
ml_strategy.py
==============

Machine-learning extension of the cross-sectional momentum strategy.

Each observation is a (date, ticker) pair.  For every such pair the features
are the per-ticker engineered signals: log returns at five horizons, rolling
volatility at two windows, rolling drawdown, and a 200-day moving-average
binary flag.  The target is whether that ticker's 21-day forward log return is
positive.

The dataset is split strictly by time: the first 70 % of dates form the
training set, the last 30 % the test set.  A StandardScaler is fitted
exclusively on training data, then applied to the test set.  A logistic
regression model is trained on the scaled training set.

At each date in the test period, the model produces a probability score for
every ticker.  Tickers are ranked cross-sectionally: the top 20 % receive a
long (+1) signal, the bottom 20 % a short (-1) signal, and the rest are flat.
Signals are scaled by the inverse of each ticker's 21-day rolling volatility
and normalised so that the sum of absolute weights equals one on each date.

Portfolio returns are computed as the daily weighted sum of next-day log
returns, giving a time series that can be evaluated with standard metrics.

Usage::

    python ml_strategy.py

Prerequisites: pandas, numpy, scikit-learn.
"""

from __future__ import annotations

import datetime as dt
import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from feature_engineering import build_features

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d",
    "vol_21d", "vol_63d", "drawdown", "above_ma_200",
]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def prepare_ml_dataset(
    prices: pd.DataFrame,
    forward_horizon: int = 21,
) -> pd.DataFrame:
    """Build a long-format (date, ticker) dataset with per-ticker features.

    Parameters
    ----------
    prices:
        Wide adjusted-close price DataFrame (dates × tickers).
    forward_horizon:
        Days ahead used to compute the binary target.

    Returns
    -------
    pandas.DataFrame
        MultiIndex (date, ticker) DataFrame containing FEATURE_COLS plus
        ``fwd_ret`` and ``target`` columns.  Rows where any feature or the
        target is NaN are dropped.
    """
    feats = build_features(prices)

    # build_features returns MultiIndex-column DataFrames:
    #   returns   → level-0 keys like "ret_1d", level-1 = ticker
    #   volatility → level-0 keys like "vol_21d", level-1 = ticker
    #   drawdown  → plain wide (dates × tickers)
    #   trend     → MultiIndex with names ["feature", "asset"],
    #               level-0 values "above_ma" / "ma_slope"
    returns = feats["returns"]
    vol     = feats["volatility"]
    dd      = feats["drawdown"]
    trend   = feats["trend"]

    fwd_ret = np.log(prices.shift(-forward_horizon) / prices)

    # Stack each wide DataFrame to a (date, ticker) Series, keeping NaNs so
    # that all series align before the final dropna.
    def to_long(wide: pd.DataFrame, name: str) -> pd.Series:
        s = wide.stack()
        s.name = name
        return s

    parts = [
        to_long(returns["ret_1d"],   "ret_1d"),
        to_long(returns["ret_5d"],   "ret_5d"),
        to_long(returns["ret_21d"],  "ret_21d"),
        to_long(returns["ret_63d"],  "ret_63d"),
        to_long(returns["ret_252d"], "ret_252d"),
        to_long(vol["vol_21d"],      "vol_21d"),
        to_long(vol["vol_63d"],      "vol_63d"),
        to_long(dd,                  "drawdown"),
        to_long(trend["above_ma"],   "above_ma_200"),
        to_long(fwd_ret,             "fwd_ret"),
    ]

    df = pd.concat(parts, axis=1)
    df.index.names = ["date", "ticker"]
    df = df.dropna(subset=FEATURE_COLS + ["fwd_ret"])
    df["target"] = (df["fwd_ret"] > 0).astype(float)
    return df


# ---------------------------------------------------------------------------
# Training and signal generation
# ---------------------------------------------------------------------------

def train_ml_strategy(prices: pd.DataFrame) -> dict:
    """Train the ML strategy and compute test-period portfolio returns.

    Parameters
    ----------
    prices:
        Wide adjusted-close price DataFrame (dates × tickers).

    Returns
    -------
    dict with keys:
        ``model``   – fitted LogisticRegression,
        ``scaler``  – fitted StandardScaler,
        ``metrics`` – performance metric dictionary,
        ``returns`` – daily portfolio return Series (test period),
        ``weights`` – wide weight DataFrame (test dates × tickers).
    """
    df = prepare_ml_dataset(prices)

    # ---- strict temporal train / test split --------------------------------
    all_dates = df.index.get_level_values("date").unique().sort_values()
    split_idx = int(len(all_dates) * 0.7)
    train_dates = all_dates[:split_idx]
    test_dates  = all_dates[split_idx:]

    in_train = df.index.get_level_values("date").isin(train_dates)
    in_test  = df.index.get_level_values("date").isin(test_dates)

    X_train = df.loc[in_train, FEATURE_COLS].to_numpy(dtype=float)
    y_train = df.loc[in_train, "target"].to_numpy(dtype=int)
    X_test  = df.loc[in_test,  FEATURE_COLS].to_numpy(dtype=float)

    # ---- scaler fitted on train only ---------------------------------------
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ---- logistic regression -----------------------------------------------
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_sc, y_train)

    # ---- cross-sectional signals on test set --------------------------------
    test_df = df.loc[in_test].copy()
    test_df["prob"] = clf.predict_proba(X_test_sc)[:, 1]

    # Rank predicted probabilities cross-sectionally within each date
    test_df["prob_rank"] = (
        test_df.groupby(level="date")["prob"]
        .rank(pct=True, method="first")
    )

    test_df["signal"] = 0.0
    test_df.loc[test_df["prob_rank"] >= 0.8, "signal"] =  1.0
    test_df.loc[test_df["prob_rank"] <= 0.2, "signal"] = -1.0

    # ---- volatility-scaled, normalised weights per date --------------------
    safe_vol = test_df["vol_21d"].replace(0.0, np.nan)
    raw_w = (test_df["signal"] / safe_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = raw_w.abs().groupby(level="date").transform("sum")
    test_df["weight"] = raw_w.where(denom == 0, raw_w / denom)

    # ---- portfolio returns: weight[t] · next-day-return[t+1] ---------------
    daily_ret = np.log(prices / prices.shift(1))

    weights_wide = test_df["weight"].unstack(level="ticker")   # dates × tickers
    ret_wide     = daily_ret.reindex(columns=weights_wide.columns)

    # shift(-1) aligns each weight row with the following day's return
    next_ret = ret_wide.shift(-1)

    common = weights_wide.index.intersection(next_ret.index)
    port_series = (
        (weights_wide.loc[common] * next_ret.loc[common])
        .sum(axis=1)
    )
    port_series.name = "portfolio_return"
    port_series.index.name = "date"

    metrics = compute_metrics(port_series)

    return {
        "model":   clf,
        "scaler":  scaler,
        "metrics": metrics,
        "returns": port_series,
        "weights": weights_wide,
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(portfolio_returns: pd.Series) -> dict:
    """Compute annualised performance metrics from a daily return series."""
    r = portfolio_returns.dropna().astype(float)
    if r.empty:
        return dict.fromkeys(
            ["cumulative_return", "annualised_return",
             "annualised_volatility", "sharpe_ratio", "max_drawdown"],
            np.nan,
        )
    n = len(r)
    cumulative  = (1 + r).prod() - 1
    ann_return  = (1 + cumulative) ** (252 / n) - 1
    ann_vol     = r.std() * np.sqrt(252)
    sharpe      = ann_return / ann_vol if ann_vol != 0 else np.nan
    cum_curve   = (1 + r).cumprod()
    max_dd      = (cum_curve / cum_curve.cummax() - 1).min()
    return {
        "cumulative_return":    cumulative,
        "annualised_return":    ann_return,
        "annualised_volatility": ann_vol,
        "sharpe_ratio":         sharpe,
        "max_drawdown":         max_dd,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    price_dir = os.path.join("data", "prices")
    files = sorted(glob.glob(os.path.join(price_dir, "adjclose_wide_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No price files found in {price_dir}. Run trade_algorithm.py first."
        )

    prices = pd.read_csv(files[-1], index_col=0, parse_dates=True)
    prices = prices.dropna(axis=1, how="all")

    result = train_ml_strategy(prices)

    print("ML strategy performance metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}")

    signals_dir = os.path.join("data", "signals")
    os.makedirs(signals_dir, exist_ok=True)
    today_str = dt.date.today().isoformat()

    returns_file = os.path.join(signals_dir, f"ml_returns_{today_str}.csv")
    result["returns"].to_csv(returns_file, index=True, header=["return"])
    print(f"Saved ML strategy returns to {returns_file}")

    weights_file = os.path.join(signals_dir, f"ml_weights_{today_str}.csv")
    result["weights"].to_csv(weights_file)
    print(f"Saved ML weights to {weights_file}")


if __name__ == "__main__":
    main()
