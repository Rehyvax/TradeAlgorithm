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
python signal_generation.py        # Generate cross-sectional momentum signals & weights
python backtester.py               # Evaluate historical performance
python robust_evaluation.py        # Grid-search parameter combinations

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
signal_generation.py    → data/signals/weights_cs_YYYY-MM-DD.csv
    ↓
daily_execution.py      → Alpaca paper API → data/live/equity_history.csv
                                           → data/live/logs/daily_execution.jsonl
```

### Key modules

- **`loaders/price_loader.py`** — `PriceDownloadConfig` dataclass and shared utilities (`download_prices_adjclose`, `quality_checks_prices`, `save_prices_panel`) used by both `trade_algorithm.py` and `signal_generation.py`.
- **`feature_engineering.py`** — Called as a library. Returns a nested dict of DataFrames keyed by feature name (log returns at multiple horizons, rolling volatility, drawdown, cross-sectional ranks). These are wide DataFrames: dates × tickers.
- **`signal_generation.py`** — Ranks assets cross-sectionally, longs top 20%, shorts bottom 20%, applies inverse-volatility scaling, outputs normalized weights (sum of abs = 1).
- **`daily_execution.py`** — The production entry point. Calls the pipeline end-to-end, enforces risk controls, converts weights to dollar notionals, and submits market orders via `alpaca.trading.client.TradingClient`. Key parameters in `main()`: `dry_run`, `gross_exposure`, `long_only`, `max_drawdown`, `target_vol`, `rebalance_band`, `min_trade_usd`.

### Risk controls (in daily_execution.py)

- **Drawdown filter**: Halts trading if cumulative drawdown from peak exceeds `max_drawdown` (default −10%).
- **Volatility scaling**: Scales down `gross_exposure` if realized volatility exceeds `target_vol` (default 10% annualized).
- **Rebalance band**: Skips position changes smaller than `rebalance_band` (default 1%) of equity to reduce turnover.
- **Cash budget**: Ensures buy orders don't exceed available cash after a safety buffer.

### Data conventions

- Prices are stored in two formats: long (`prices_long_YYYY-MM-DD.csv`) and wide (`adjclose_wide_YYYY-MM-DD.csv`).
- Signals and weights are date-stamped: `signals_cs_YYYY-MM-DD.csv`, `weights_cs_YYYY-MM-DD.csv`.
- Weights are lagged by one day when used in backtests (avoids look-ahead bias).
- The trading universe is 24 ETFs defined in `data/universe.csv`; assets are filtered by minimum $500k daily volume and $200M AUM at ingestion time.
- Audit trail for live runs is in `data/live/logs/daily_execution.jsonl` (JSONL format).
