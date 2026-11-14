# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full daily pipeline (data → signals → orders)
python daily_execution.py          # dry_run=True by default (no real orders)

# Individual pipeline steps
python trade_algorithm.py          # Download & validate prices from Yahoo Finance
python trend_strategy.py           # Core strategy: backtest, benchmarks, robustness table
python thematic_sleeve.py          # Thematic sleeve vs buy-and-hold basket (downloads stock prices)
python short_horizon_study.py      # Short-window momentum / reversal study (downloads prices)

# Legacy research (original cross-sectional strategy)
python signal_generation.py        # Generate cross-sectional momentum signals & weights
python backtester.py               # Evaluate historical performance
python robust_evaluation.py        # Grid-search parameter combinations
python walk_forward.py             # Walk-forward validation
python universe_segments.py        # Segmented cross-sectional momentum

# Utilities
python test_alpaca_connection.py   # Verify Alpaca API credentials
python live_dashboard.py           # Plot equity curve from trading history

# Scheduled execution (Windows Task Scheduler)
run_daily.bat                      # Runs daily_execution.py via .venv, logs to data/live/cron.log
```

**Configuration**: Alpaca credentials go in a `.env` file as `ALPACA_API_KEY` and `ALPACA_API_SECRET`.

There is no test suite or linter configured.

## Architecture

The pipeline is **filesystem-coupled**: each module produces timestamped CSV files that the next module reads. Steps can be run independently using the latest file.

```
Yahoo Finance
    ↓
trade_algorithm.py      → data/prices/adjclose_wide_YYYY-MM-DD.csv
    ↓
feature_engineering.py  → (imported as a library, not run directly)
    ↓
trend_strategy.py       → core sleeve weights (ETF universe)
thematic_sleeve.py      → thematic sleeve weights (data/universe_thematic.csv)
risk_management.py      → exposure multiplier from account equity
    ↓
daily_execution.py      → Alpaca paper API → data/live/equity_history.csv
                                           → data/live/logs/daily_execution.jsonl
```

### Key modules

- **`loaders/price_loader.py`**: `PriceDownloadConfig` dataclass and shared utilities (`download_prices_adjclose`, `quality_checks_prices`, `save_prices_panel`).
- **`trend_strategy.py`**: Core strategy. `TrendConfig` dataclass; `run_trend_strategy` for backtests and `latest_target_weights` for execution. Long or flat blended trend (63/126/252 days), relative-strength tilt inside each sleeve, risk budgeting, 10% volatility target capped at 100% gross, staggered weekly rebalance, 2% no-trade band. Uninvested capital goes to `SHY`.
- **`thematic_sleeve.py`**: Volatility scaling and trend gate on the stocks listed in `data/universe_thematic.csv`. `run_thematic_sleeve` for backtests and `latest_sleeve_weights` for execution.
- **`risk_management.py`**: Pure functions. `compute_exposure(equity_series, RiskConfig)` returns the exposure multiplier from equity sanitisation, current drawdown, realised volatility and the kill-switch.
- **`performance_metrics.py`**: Reference metrics for the new modules (Sharpe over a risk-free series, Deflated Sharpe Ratio, bootstrap interval). The legacy modules keep their own metric functions.
- **`short_horizon_study.py`**: Research only. Tests k-day momentum and reversal rules with next-open execution.
- **`daily_execution.py`**: The production entry point. `strategy="trend"` (default) combines the core sleeve (70%) and the thematic sleeve (30%) in `build_trend_targets`; `strategy="legacy"` reproduces the original segmented cross-sectional behaviour. Key parameters in `main()`: `dry_run`, `gross_exposure`, `max_drawdown`, `target_vol`, `rebalance_band`, `min_trade_usd`, `strategy`, `thematic_allocation`.
- **`feature_engineering.py`, `signal_generation.py`, `universe_segments.py`, `walk_forward.py`, `robust_evaluation.py`, `backtester.py`, `ml_strategy.py`**: Legacy cross-sectional momentum research, kept for reference and comparison.

### Risk controls (in risk_management.py, applied by daily_execution.py)

- **Equity sanitisation**: discards equity prints that jump more than 15% and revert; an unconfirmed jump today means no orders today.
- **Drawdown scaling**: based on the *current* drawdown from the 252-run peak; exposure falls from 100% at `max_drawdown / 2` to 35% at `max_drawdown` and recovers with the account.
- **Kill-switch**: at `2 * max_drawdown` the account moves to cash for 21 runs, then re-arms.
- **Volatility scaling**: scales exposure down if realised volatility exceeds `target_vol`.
- **Rebalance band**: skips position changes smaller than `rebalance_band` (default 2%) of equity.
- **Cash budget**: ensures buy orders don't exceed available cash after a safety buffer.

### Data conventions

- Prices are stored in two formats: long (`prices_long_YYYY-MM-DD.csv`) and wide (`adjclose_wide_YYYY-MM-DD.csv`).
- Signals and weights are date-stamped: `weights_live_YYYY-MM-DD.csv` (trend strategy, what execution trades), `weights_trend_*.csv`, `weights_thematic_*.csv`, and the legacy `signals_cs_*.csv`, `weights_cs_*.csv`, `weights_segmented_*.csv`.
- Weights are lagged by one day when used in backtests (avoids look-ahead bias).
- The trading universe is 24 ETFs defined in `data/universe.csv`; assets are filtered by minimum $500k daily volume and $200M AUM at ingestion time.
- Audit trail for live runs is in `data/live/logs/daily_execution.jsonl` (JSONL format).
