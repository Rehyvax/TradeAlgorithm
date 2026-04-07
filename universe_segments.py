"""
universe_segments.py
====================

Segmented cross-sectional momentum strategy.

The core idea is that ranking assets *within* homogeneous groups preserves
economic meaning: a commodity ETF in the top decile of commodities is a
genuine momentum signal; the same ETF in the top decile of a mixed universe
that includes government bonds is not, because the two assets respond to
completely different fundamental drivers.

This module implements:

  run_segmented_strategy()
      Apply cross-sectional momentum independently within each segment,
      combine with equal group-level weights, and normalise.

  run_segmented_walk_forward()
      Same 5-year train / 1-year test rolling windows as walk_forward.py,
      but using run_segmented_strategy instead of run_strategy.

  main()
      Load prices, run segmented walk-forward, print per-window and overall
      metrics, save CSVs, and print a side-by-side comparison with the
      original walk-forward results.

Usage::

    python universe_segments.py
"""

from __future__ import annotations

import datetime as dt
import glob
import os
from itertools import product as iproduct

import numpy as np
import pandas as pd

from robust_evaluation import compute_metrics, run_strategy

# ---------------------------------------------------------------------------
# Universe segmentation
# ---------------------------------------------------------------------------

SEGMENTS: dict[str, list[str]] = {
    "equity":       ["EEM", "EFA", "IWD", "IWF", "IWN", "IWO", "SCZ"],
    "fixed_income": ["BWX", "EMB", "HYG", "LQD", "SHY", "TIP", "TLT"],
    "alternatives": ["GLD", "USO", "BIZD", "BKLN"],
}

# Reduced grid reused from walk_forward.py
RANKING_HORIZONS = [63, 126, 252]
TOP_PCTS         = [0.1, 0.2]
VOL_WINDOWS      = [21, 63]


# ---------------------------------------------------------------------------
# Segmented strategy (single period)
# ---------------------------------------------------------------------------

def run_segmented_strategy(
    prices: pd.DataFrame,
    ranking_horizon: int = 252,
    top_pct: float = 0.3,
    bottom_pct: float = 0.3,
    vol_window: int = 63,
) -> dict:
    """Apply cross-sectional momentum within each segment and combine.

    Parameters
    ----------
    prices:
        Wide adjusted-close price DataFrame (dates x tickers).
    ranking_horizon:
        Lookback window in trading days for momentum ranking.
    top_pct:
        Fraction of assets to go long within each segment.
    bottom_pct:
        Fraction of assets to go short within each segment.
    vol_window:
        Rolling window for volatility scaling.

    Returns
    -------
    dict
        Same structure as ``run_strategy``: keys ``metrics``, ``weights``,
        ``returns``.  ``metrics`` includes an extra ``"segments_used"`` key.
    """
    n_groups = len(SEGMENTS)
    group_weight_share = 1.0 / n_groups  # each group contributes equally

    combined_weights: pd.DataFrame | None = None

    segments_used = []
    for segment_name, tickers in SEGMENTS.items():
        # Keep only tickers present in this price panel
        available = [t for t in tickers if t in prices.columns]
        if len(available) < 2:
            # Need at least 2 assets to form a long/short within the segment
            continue

        seg_prices = prices[available]
        seg_result = run_strategy(seg_prices, ranking_horizon, top_pct, bottom_pct, vol_window)
        seg_weights = seg_result["weights"]  # dates x available_tickers, sum(|w|)=1 per date

        # Scale so this segment contributes exactly group_weight_share to
        # the combined portfolio (regardless of how many tickers it has).
        seg_weights_scaled = seg_weights * group_weight_share

        if combined_weights is None:
            combined_weights = seg_weights_scaled
        else:
            # Align on the union of dates; missing dates get 0 weight
            combined_weights = combined_weights.add(seg_weights_scaled, fill_value=0.0)

        segments_used.append(segment_name)

    if combined_weights is None or combined_weights.empty:
        raise ValueError("No segment had enough tickers to run. Check SEGMENTS and prices.")

    # Re-normalise so sum(|w|) = 1 on each date (some dates may have fewer
    # active segments if data starts later for some assets)
    denom = combined_weights.abs().sum(axis=1).replace(0.0, np.nan)
    weights_norm = combined_weights.div(denom, axis=0).fillna(0.0)

    # Portfolio returns: weights lagged by one day, applied to log returns
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    w = weights_norm.reindex(log_returns.index).fillna(0.0)
    gross = (w.shift(1).fillna(0.0) * log_returns.reindex(columns=w.columns).fillna(0.0)).sum(axis=1)

    # Flat 5 bps transaction cost on turnover
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    port_returns = gross - turnover * (5.0 / 10_000.0)

    metrics = compute_metrics(port_returns)
    metrics["segments_used"] = ", ".join(segments_used)

    return {
        "metrics": metrics,
        "weights": weights_norm,
        "returns": port_returns,
    }


# ---------------------------------------------------------------------------
# Segmented walk-forward
# ---------------------------------------------------------------------------

def run_segmented_walk_forward(
    prices: pd.DataFrame,
    n_train_years: int = 5,
    n_test_years:  int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Walk-forward validation using the segmented momentum strategy.

    Reuses the same window schedule as ``walk_forward.run_walk_forward``:
    a ``n_train_years``-year training window to select parameters via grid
    search, followed by a ``n_test_years``-year out-of-sample test.  The
    window advances by ``n_test_years`` each iteration.

    Parameters
    ----------
    prices:
        Wide adjusted-close price DataFrame (dates x tickers).
    n_train_years:
        Length of the training window in years.
    n_test_years:
        Length of the out-of-sample test window in years.

    Returns
    -------
    results_df : pd.DataFrame
        One row per window with columns: ``window_start``,
        ``window_test_start``, ``window_end``, ``sharpe_ratio``,
        ``annualised_return``, ``max_drawdown``, ``best_ranking_horizon``,
        ``best_top_pct``, ``best_vol_window``.
    oos_returns : pd.Series
        Concatenated daily OOS returns across all test windows.
    """
    prices = prices.sort_index().dropna(axis=1, how="all")
    first_date = prices.index[0]
    last_date  = prices.index[-1]

    # ── Build window schedule (identical to walk_forward.py) ──────────────
    windows: list[tuple] = []
    test_start = first_date + pd.DateOffset(years=n_train_years)
    while True:
        test_end = test_start + pd.DateOffset(years=n_test_years)
        if test_end > last_date:
            break
        train_start = test_start - pd.DateOffset(years=n_train_years)
        windows.append((train_start, test_start, test_end))
        test_start = test_end

    if not windows:
        raise ValueError(
            f"Not enough history for {n_train_years}-year train + "
            f"{n_test_years}-year test windows."
        )

    print(
        f"Segmented walk-forward: {len(windows)} windows | "
        f"train={n_train_years}y, test={n_test_years}y | "
        f"grid size={len(RANKING_HORIZONS) * len(TOP_PCTS) * len(VOL_WINDOWS)} combos/segment | "
        f"{len(SEGMENTS)} segments"
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
            print(f"  [{i+1}/{len(windows)}] SKIP -- only {n_train_rows} train rows")
            continue
        if n_test_rows == 0:
            print(f"  [{i+1}/{len(windows)}] SKIP -- no test rows in range")
            continue

        # ── Grid search on TRAIN data ──────────────────────────────────────
        # Evaluate run_segmented_strategy across parameter combinations and
        # pick the set that maximises Sharpe on the training window.
        best_sharpe_train = -np.inf
        best_params: tuple | None = None

        for rh, pct, vw in iproduct(RANKING_HORIZONS, TOP_PCTS, VOL_WINDOWS):
            if n_train_rows < rh + vw:
                continue
            try:
                res = run_segmented_strategy(train_prices, rh, pct, pct, vw)
                s   = res["metrics"]["sharpe_ratio"]
            except Exception:
                continue
            if pd.notna(s) and s > best_sharpe_train:
                best_sharpe_train = s
                best_params = (rh, pct, vw)

        if best_params is None:
            print(f"  [{i+1}/{len(windows)}] SKIP -- no valid param combo on train data")
            continue

        rh, pct, vw = best_params

        # ── Evaluate on TEST data with warm-up ────────────────────────────
        eval_prices = prices.loc[prices.index < test_end]
        test_res    = run_segmented_strategy(eval_prices, rh, pct, pct, vw)

        actual_test_start = prices.loc[test_mask].index[0]
        actual_test_end   = prices.loc[test_mask].index[-1]
        oos_ret = test_res["returns"].loc[actual_test_start:actual_test_end]

        oos_metrics = compute_metrics(oos_ret)

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
            "window_start":         actual_train_start.date(),
            "window_test_start":    actual_test_start.date(),
            "window_end":           actual_test_end.date(),
            "sharpe_ratio":         round(oos_metrics["sharpe_ratio"],     4),
            "annualised_return":    round(oos_metrics["annualised_return"], 4),
            "max_drawdown":         round(oos_metrics["max_drawdown"],     4),
            "best_ranking_horizon": rh,
            "best_top_pct":         pct,
            "best_vol_window":      vw,
        })
        oos_chunks.append(oos_ret)

    results_df  = pd.DataFrame(results)
    oos_returns = pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float, name="return")

    return results_df, oos_returns


# ---------------------------------------------------------------------------
# Side-by-side comparison helper
# ---------------------------------------------------------------------------

def _compare_results(
    segmented_df: pd.DataFrame,
    original_csv_pattern: str,
) -> None:
    """Print a side-by-side Sharpe comparison against the original walk-forward."""
    orig_files = sorted(glob.glob(original_csv_pattern))
    if not orig_files:
        print("(Original walk-forward results not found — skipping comparison)")
        return

    orig_df = pd.read_csv(orig_files[-1])
    orig_df["window_test_start"] = orig_df["window_test_start"].astype(str)
    seg_df  = segmented_df.copy()
    seg_df["window_test_start"] = seg_df["window_test_start"].astype(str)

    merged = pd.merge(
        orig_df[["window_test_start", "sharpe_ratio"]].rename(columns={"sharpe_ratio": "sharpe_original"}),
        seg_df[["window_test_start",  "sharpe_ratio"]].rename(columns={"sharpe_ratio": "sharpe_segmented"}),
        on="window_test_start",
        how="outer",
    ).sort_values("window_test_start")

    merged["delta"] = merged["sharpe_segmented"] - merged["sharpe_original"]
    merged["better"] = merged["delta"].apply(lambda d: "YES" if pd.notna(d) and d > 0 else ("--" if pd.isna(d) else "no"))

    print()
    print("Side-by-side Sharpe comparison (original vs segmented)")
    print(f"{'window_test_start':<20} {'sharpe_original':>16} {'sharpe_segmented':>17} {'delta':>8} {'better?':>9}")
    print("-" * 74)
    for _, row in merged.iterrows():
        orig_s = f"{row['sharpe_original']:+.3f}" if pd.notna(row["sharpe_original"]) else "    N/A"
        seg_s  = f"{row['sharpe_segmented']:+.3f}" if pd.notna(row["sharpe_segmented"]) else "    N/A"
        delta  = f"{row['delta']:+.3f}" if pd.notna(row["delta"]) else "    N/A"
        print(f"{str(row['window_test_start']):<20} {orig_s:>16} {seg_s:>17} {delta:>8} {row['better']:>9}")
    print("-" * 74)

    n_better = (merged["delta"] > 0).sum()
    n_total  = merged["delta"].notna().sum()
    avg_orig = merged["sharpe_original"].mean()
    avg_seg  = merged["sharpe_segmented"].mean()
    print(f"  Segmented better in {n_better}/{n_total} windows")
    print(f"  Mean Sharpe -- original: {avg_orig:+.3f}  segmented: {avg_seg:+.3f}  delta: {avg_seg - avg_orig:+.3f}")


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

    print(f"Loaded: {len(prices)} rows x {len(prices.columns)} tickers")
    print(f"Range : {prices.index[0].date()} -- {prices.index[-1].date()}")
    print()
    print("Segments:")
    for name, tickers in SEGMENTS.items():
        available = [t for t in tickers if t in prices.columns]
        print(f"  {name:<15} {len(available)}/{len(tickers)} tickers available: {available}")
    print()

    results_df, oos_returns = run_segmented_walk_forward(prices)

    print()
    print("=" * 60)

    if not oos_returns.empty:
        overall = compute_metrics(oos_returns)
        print("Overall OOS performance (segmented, all windows concatenated):")
        for k, v in overall.items():
            print(f"  {k}: {v:.4f}")
        print()

    if not results_df.empty:
        print("Per-window summary (segmented):")
        print(
            results_df[["window_test_start", "window_end",
                         "sharpe_ratio", "annualised_return",
                         "max_drawdown"]].to_string(index=False)
        )

    # ── Save outputs ───────────────────────────────────────────────────────
    signals_dir = os.path.join("data", "signals")
    os.makedirs(signals_dir, exist_ok=True)
    today_str = dt.date.today().isoformat()

    results_file = os.path.join(signals_dir, f"segmented_walk_forward_results_{today_str}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved window results to {results_file}")

    returns_file = os.path.join(signals_dir, f"segmented_walk_forward_returns_{today_str}.csv")
    oos_returns.to_csv(returns_file, index=True, header=["return"])
    print(f"Saved OOS returns    to {returns_file}")

    # ── Side-by-side comparison ────────────────────────────────────────────
    original_pattern = os.path.join(signals_dir, "walk_forward_results_*.csv")
    _compare_results(results_df, original_pattern)


if __name__ == "__main__":
    main()
