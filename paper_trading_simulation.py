"""
paper_trading_simulation.py
==========================

This example script illustrates how you might connect a
research strategy to a paper‑trading API in order to
simulate order generation and position management with
fictitious capital.  It uses the Alpaca API as an
example broker; you can adapt it to other services
that offer paper trading (e.g. Interactive Brokers,
IBKR) by modifying the API calls accordingly.

**Important:** This script does not send any orders
unless you provide valid API credentials and remove
the early return.  It is provided for educational
purposes and requires careful adaptation before use
in production.  You must comply with all applicable
regulations and your broker's terms of service.

Usage::

    export ALPACA_API_KEY="your_key"
    export ALPACA_API_SECRET="your_secret"
    python TradeAlgorithm/paper_trading_simulation.py

Design Notes
------------

The script performs the following steps:

1. Connects to Alpaca's paper trading REST API.
2. Loads the most recent price panel and computes the
   best‑performing parameter set from the robust
   evaluation (or uses a fixed set).
3. Calculates target portfolio weights and translates
   them into share quantities based on current
   account equity and last prices.
4. Compares current positions with target positions
   and generates orders to trade the difference.

You should run this script once per day after
market close or at a scheduled time.  It does not
operate continuously.
"""

import os
import glob
import pandas as pd
import numpy as np

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

# Import run_strategy from robust_evaluation for signal generation
from robust_evaluation import run_strategy


def load_latest_prices(price_dir: str) -> pd.DataFrame:
    pattern = os.path.join(price_dir, "adjclose_wide_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No price files found in {price_dir}. Run trade_algorithm.py first."
        )
    latest_file = files[-1]
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    return df


def compute_target_weights(prices: pd.DataFrame) -> pd.Series:
    """Compute desired portfolio weights using the best
    performing configuration from the last robust evaluation.

    This function reads the results of the robust grid
    search and returns the last row (highest Sharpe) and
    uses those parameters to run the strategy and extract
    the most recent weights.
    """
    # Load robust results
    signals_dir = os.path.join("data", "signals")
    results_files = sorted(
        glob.glob(os.path.join(signals_dir, "robust_results_*.csv"))
    )
    if not results_files:
        raise FileNotFoundError(
            "Robust results file not found. Run robust_evaluation.py first."
        )
    latest_results = pd.read_csv(results_files[-1])
    # Select best row
    best_row = latest_results.sort_values(
        by="sharpe_ratio", ascending=False
    ).iloc[0]
    # Run strategy with best parameters
    res = run_strategy(
        prices,
        ranking_horizon=int(best_row["ranking_horizon"]),
        top_pct=float(best_row["top_pct"]),
        bottom_pct=float(best_row["bottom_pct"]),
        vol_window=int(best_row["vol_window"]),
    )
    # Take most recent weights
    weights = res["weights"].iloc[-1]
    return weights


def alpaca_paper_trade(weights: pd.Series):
    """Send orders to Alpaca paper trading based on target weights.

    Parameters
    ----------
    weights : pandas.Series
        Target weights indexed by ticker symbol.
    """
    if tradeapi is None:
        raise ImportError(
            "alpaca_trade_api is not installed. Please install it with `pip install alpaca-trade-api`."
        )
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )
    if not api_key or not api_secret:
        raise EnvironmentError(
            "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
        )
    api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
    # Fetch current account equity
    account = api.get_account()
    equity = float(account.equity)
    # Fetch last price for each symbol
    symbols = weights.index.tolist()
    quotes = api.get_latest_quotes(symbols)
    prices = {q.symbol: float(q.ask_price) for q in quotes}
    # Compute target dollar exposure per symbol
    target_dollars = weights * equity
    # Determine current positions
    current_positions = {p.symbol: float(p.qty) for p in api.list_positions()}
    orders_to_submit = []
    for sym, w in weights.items():
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        target_shares = target_dollars[sym] / price
        current_shares = current_positions.get(sym, 0.0)
        diff = target_shares - current_shares
        # Round to nearest share (can adjust to lot size)
        diff_int = int(np.floor(diff)) if diff > 0 else int(np.ceil(diff))
        if diff_int != 0:
            side = "buy" if diff_int > 0 else "sell"
            qty = abs(diff_int)
            orders_to_submit.append({
                "symbol": sym,
                "qty": qty,
                "side": side,
                "type": "market",
                "time_in_force": "day",
            })
    # Print proposed trades
    print("Proposed trades:")
    for o in orders_to_submit:
        print(o)
    # Uncomment the following lines to actually submit orders.
    # for o in orders_to_submit:
    #     api.submit_order(**o)
    print(
        "Orders not submitted (paper trading). Remove the comment to submit to the broker."
    )


def main():
    price_dir = os.path.join("data", "prices")
    prices = load_latest_prices(price_dir)
    # Compute target weights
    weights = compute_target_weights(prices)
    # Print weights
    print("Target weights (sum to 1):")
    print(weights)
    # Connect to Alpaca and generate trades
    try:
        alpaca_paper_trade(weights)
    except Exception as e:
        print(f"Paper trading error: {e}")


if __name__ == "__main__":
    main()