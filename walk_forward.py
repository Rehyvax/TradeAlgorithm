"""
walk_forward.py
===============

Walk-forward validation for the cross-sectional momentum strategy.

Instead of fitting parameters on the full historical dataset (in-sample),
this module divides the price history into rolling windows:

  - **Train** window  : ``n_train_years`` years, used to find the best
    parameter combination via grid search (Sharpe ratio on train data).
  - **Test** window   : ``n_test_years`` years immediately after train,
    used to evaluate the strategy out-of-sample with the winning params.
  - The window then advances by ``n_test_years`` and the process repeats.

With 20+ years of data (2005 – present), a 5-year train / 1-year test
split yields ~15-16 non-overlapping test periods.

The reduced parameter grid (3 × 2 × 2 = 12 combinations) keeps each
window's grid search fast enough to run interactively.

Usage::

    python walk_forward.py

Outputs (written to ``data/signals/``):
  - ``walk_forward_results_YYYY-MM-DD.csv``  — one row per test window
  - ``walk_forward_returns_YYYY-MM-DD.csv``  — concatenated OOS daily returns
"""

from __future__ import annotations

import datetime as dt
import glob
import os
from itertools import product as iproduct

import numpy as np
import pandas as pd

from robust_evaluation import compute_metrics, run_strategy

# Reduced grid for walk-forward (12 combinations per window)
RANKING_HORIZONS = [63, 126, 252]
TOP_PCTS         = [0.1, 0.2]
VOL_WINDOWS      = [21, 63]


# ---------------------------------------------------------------------------
# Core walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    prices: pd.DataFrame,
    n_train_years: int = 5,
    n_test_years:  int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run walk-forward parameter selection and out-of-sample evaluation.

    Parameters
    ----------
    prices:
        Wide adjusted-close price DataFrame (dates × tickers), sorted by date.
    n_train_years:
        Length of the training window in years.
    n_test_years:
        Length of the test (out-of-sample) window in years. Also the step
        size by which the window advances each iteration.

    Returns
    -------
    results_df : pd.DataFrame
        One row per test window with columns:
        ``window_start``, ``window_test_start``, ``window_end``,
        ``sharpe_ratio``, ``annualised_return``, ``max_drawdown``,
        ``best_ranking_horizon``, ``best_top_pct``, ``best_vol_window``.
    oos_returns : pd.Series
        Concatenated daily out-of-sample returns across all test windows.
    """
    prices = prices.sort_index().dropna(axis=1, how="all")
    first_date = prices.index[0]
    last_date  = prices.index[-1]

    # ── Build window schedule ──────────────────────────────────────────────
    windows: list[tuple] = []
    test_start = first_date + pd.DateOffset(years=n_train_years)
    while True:
        test_end = test_start + pd.DateOffset(years=n_test_years)
        if test_end > last_date:
            break
        train_start = test_start - pd.DateOffset(years=n_train_years)
        windows.append((train_start, test_start, test_end))
        test_start = test_end  # advance by one test period (non-overlapping OOS)

    if not windows:
        raise ValueError(
            f"Not enough history for {n_train_years}-year train + "
            f"{n_test_years}-year test windows. "
            f"Data spans {first_date.date()} – {last_date.date()}."
        )

    print(
        f"Walk-forward: {len(windows)} windows | "
        f"train={n_train_years}y, test={n_test_years}y | "
        f"grid size={len(RANKING_HORIZONS) * len(TOP_PCTS) * len(VOL_WINDOWS)} combos"
    )
    print()

    results: list[dict] = []
    oos_chunks: list[pd.Series] = []

    for i, (train_start, test_start, test_end) in enumerate(windows):

        train_mask = (prices.index >= train_start) & (prices.index < test_start)
        test_mask  = (prices.index >= test_start)  & (prices.index < test_end)

        train_prices = prices.loc[train_mask]
        n_train_rows = train_mask.sum()
        n_test_rows  = test_mask.sum()

        if n_train_rows < 300:
            print(f"  [{i+1}/{len(windows)}] SKIP — only {n_train_rows} train rows")
            continue
        if n_test_rows == 0:
            print(f"  [{i+1}/{len(windows)}] SKIP — no test rows in range")
            continue

        # ── Grid search on TRAIN data ──────────────────────────────────────
        best_sharpe_train = -np.inf
        best_params:  tuple | None = None

        for rh, pct, vw in iproduct(RANKING_HORIZONS, TOP_PCTS, VOL_WINDOWS):
            if n_train_rows < rh + vw:
                continue  # not enough rows for this combination
            res = run_strategy(train_prices, rh, pct, pct, vw)
            s   = res["metrics"]["sharpe_ratio"]
            if pd.notna(s) and s > best_sharpe_train:
                best_sharpe_train = s
                best_params = (rh, pct, vw)

        if best_params is None:
            print(f"  [{i+1}/{len(windows)}] SKIP — no valid combo found on train data")
            continue

        rh, pct, vw = best_params

        # ── Evaluate on TEST data with warm-up from train period ───────────
        # Pass all prices up to the end of the test window so that
        # run_strategy has enough lookback to compute momentum/vol at the
        # very first test date. We then keep only the test-period returns.
        eval_prices = prices.loc[prices.index < test_end]
        test_res    = run_strategy(eval_prices, rh, pct, pct, vw)
        # Filter to test period using dates, not the full-length boolean mask
        actual_test_start = prices.loc[test_mask].index[0]
        actual_test_end   = prices.loc[test_mask].index[-1]
        oos_ret = test_res["returns"].loc[actual_test_start:actual_test_end]

        oos_metrics = compute_metrics(oos_ret)

        # Actual boundary dates (nearest available trading days)
        actual_train_start = prices.loc[train_mask].index[0]
        actual_train_end   = prices.loc[train_mask].index[-1]

        print(
            f"  [{i+1}/{len(windows)}] "
            f"train {actual_train_start.date()} -> {actual_train_end.date()}  |  "
            f"test {actual_test_start.date()} -> {actual_test_end.date()}  |  "
            f"params: horizon={rh:3d}, top={pct:.0%}, vol_win={vw:2d}  |  "
            f"OOS Sharpe={oos_metrics['sharpe_ratio']:+.3f}"
        )

        results.append({
            "window_start":          actual_train_start.date(),
            "window_test_start":     actual_test_start.date(),
            "window_end":            actual_test_end.date(),
            "sharpe_ratio":          round(oos_metrics["sharpe_ratio"],    4),
            "annualised_return":     round(oos_metrics["annualised_return"], 4),
            "max_drawdown":          round(oos_metrics["max_drawdown"],    4),
            "best_ranking_horizon":  rh,
            "best_top_pct":          pct,
            "best_vol_window":       vw,
        })
        oos_chunks.append(oos_ret)

    results_df  = pd.DataFrame(results)
    oos_returns = pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float, name="return")

    return results_df, oos_returns


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

    print(f"Loaded: {len(prices)} rows × {len(prices.columns)} tickers")
    print(f"Range : {prices.index[0].date()} – {prices.index[-1].date()}")
    print()

    results_df, oos_returns = run_walk_forward(prices)

    print()
    print("=" * 60)

    if not oos_returns.empty:
        overall = compute_metrics(oos_returns)
        print("Overall OOS performance (all windows concatenated):")
        for k, v in overall.items():
            print(f"  {k}: {v:.4f}")
        print()

    if not results_df.empty:
        print("Per-window summary:")
        print(results_df[["window_test_start", "window_end",
                           "sharpe_ratio", "annualised_return",
                           "max_drawdown"]].to_string(index=False))
        print()

    signals_dir = os.path.join("data", "signals")
    os.makedirs(signals_dir, exist_ok=True)
    today_str = dt.date.today().isoformat()

    results_file = os.path.join(signals_dir, f"walk_forward_results_{today_str}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Saved window results to {results_file}")

    returns_file = os.path.join(signals_dir, f"walk_forward_returns_{today_str}.csv")
    oos_returns.to_csv(returns_file, index=True, header=["return"])
    print(f"Saved OOS returns    to {returns_file}")


if __name__ == "__main__":
    main()
