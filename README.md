# TradeAlgorithm

![Paper Trading](https://img.shields.io/badge/mode-paper%20trading-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-informational)
![License](https://img.shields.io/badge/license-MIT-green)

**An end-to-end systematic trading research pipeline** — from raw market data to signal generation, walk-forward validation, ML extensions, and automated paper-trading execution via Alpaca.

---

## Overview

TradeAlgorithm implements a **cross-sectional momentum strategy** on a universe of 24 ETFs spanning global equities, fixed income, commodities, and alternatives. The pipeline handles the full research lifecycle: downloading and validating price data, engineering features, generating volatility-scaled signals, backtesting with realistic transaction costs, and submitting orders to Alpaca's paper trading environment on a daily schedule.

The project is designed as a rigorous research prototype rather than a production system. It includes a walk-forward validation framework to produce honest out-of-sample estimates, an ML extension using logistic regression for signal generation, and a web dashboard for live monitoring. Results — including negative findings — are documented transparently.

---

## Strategy

### Cross-Sectional Momentum

Cross-sectional momentum (Jegadeesh & Titman, 1993) is the observation that assets which have outperformed their peers over the past 6–12 months tend to continue outperforming over the next 1–3 months, and vice versa. The theoretical justification combines behavioural finance (investor underreaction to information, herding) and risk-based explanations (momentum as compensation for crash risk during periods of market stress).

The implementation follows a four-step construction:

```
1. RANK    — compute each asset's log return over the lookback window
             rank cross-sectionally (percentile within the universe, each date)

2. SIGNAL  — long (+1) the top 20 % of ranked assets
             short (−1) the bottom 20 %
             flat (0) the middle 60 %

3. VOL-SCALE — divide each signal by the asset's rolling volatility
               (21- or 63-day std of daily log returns)
               → positions sized inversely to risk, not notional value

4. NORMALIZE — scale all weights so Σ|wᵢ| = 1
               → gross exposure is controlled regardless of how many signals fire
```

Risk controls layered on top of the signal include a maximum drawdown circuit breaker (−10%), a volatility targeting overlay (10% annualised target), and a rebalance band (1% minimum position change) to reduce unnecessary turnover.

---

## Pipeline

```
Yahoo Finance (24 ETFs, daily OHLCV, from 2005)
        │
        ▼
┌─────────────────────┐
│  trade_algorithm.py │  Download & validate prices, filter by liquidity
│  loaders/           │  Save adjclose_wide, prices_long, volume_wide CSVs
└─────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  feature_engineering.py  │  Log returns (1d/5d/21d/63d/252d), rolling vol,
│                          │  drawdown, 200-day MA trend flag, CS ranks
└──────────────────────────┘
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
┌────────────────────┐                       ┌─────────────────────┐
│ signal_generation  │  CS momentum signals  │   ml_strategy.py    │
│       .py          │  + vol-scaled weights │                     │
│                    │  → data/signals/      │  Logistic regression│
└────────────────────┘                       │  on per-(date,tkr)  │
        │                                    │  feature rows; OOS  │
        │                                    │  signals by CS rank │
        │                                    └─────────────────────┘
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│ backtester   │   │ robust_evaluation │   │   walk_forward.py    │
│    .py       │   │      .py          │   │                      │
│              │   │                   │   │ 5-year train /       │
│ Historical   │   │ Grid search over  │   │ 1-year test windows  │
│ performance  │   │ (horizon, pct,    │   │ Honest OOS estimates │
│ + market-    │   │  vol_window)      │   │ of strategy value    │
│ impact costs │   │ + market-impact   │   └──────────────────────┘
└──────────────┘   └───────────────────┘
        │
        ▼
┌──────────────────────┐
│  daily_execution.py  │  Risk controls → Alpaca Paper API → orders
│                      │  Logs → data/live/logs/daily_execution.jsonl
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ dashboard_server.py  │  Flask web UI — equity curve, positions,
│ templates/           │  robustness table, execution logs
│   dashboard.html     │  http://localhost:5050
└──────────────────────┘
```

---

## Research Findings

> **This section documents the walk-forward validation results honestly. Transparency about what works and what does not is a prerequisite for sound quantitative research.**

### Walk-Forward Results (16 windows, 2010–2026)

Walk-forward validation was performed with a 5-year training window and a 1-year out-of-sample test window, advancing annually. Parameters were selected by Sharpe ratio on the training data alone and applied without modification to the test period.

| Metric | Value |
|---|---|
| OOS Sharpe ratio | **−0.11** |
| Annualised return (OOS) | **−1.2 %** |
| Max drawdown (OOS) | **−39.7 %** |
| Positive windows | **6 of 16** |
| Best window | **2013 (Sharpe +1.70)** |
| Worst window | **2016 (Sharpe −1.83)** |

The dominant parameter selected during training shifted over time: `horizon=126` (6-month lookback) dominated before 2020; `horizon=252` (12-month lookback) won consistently from 2020 onwards. This parameter instability is itself informative — it indicates that no single lookback period is stable across market regimes.

### Universe Segmentation — Hypothesis Validation

To test Hypothesis 1 directly, the strategy was re-run using `universe_segments.py`, which applies cross-sectional momentum *independently within* each of three homogeneous groups (equity, fixed income, alternatives) and combines the resulting signals with equal group-level weights. Parameters were selected by the same walk-forward procedure.

| Metric | Original | Segmented | Delta |
|---|---|---|---|
| OOS Sharpe ratio | −0.11 | **+0.31** | **+0.42** |
| Annualised return | −1.2% | **+3.3%** | **+4.5 pp** |
| Max drawdown | −39.7% | **−24.3%** | **+15.4 pp** |
| Positive windows | 6 of 16 | **11 of 16** | **+5** |

The improvement confirms the hypothesis empirically: momentum is a real and persistent signal *within* homogeneous asset classes, but ranking across fundamentally different asset classes — comparing a commodity ETF momentum signal against a government bond momentum signal — destroys the factor by conflating unrelated economic mechanisms. The cross-sectional rank loses its meaning when the assets being ranked do not share a common return driver.

The only windows where segmentation underperforms are 2013 (delta −1.12) and 2015 (delta −1.64) — precisely the windows where the original strategy already had a strong global signal. This is consistent with the interpretation that segmentation adds a constraint (rank only within groups) that is beneficial when the global signal is noisy, but introduces unnecessary friction when the global signal was already clean and informative.

### Hypotheses

The negative aggregate OOS performance can be attributed to three structural factors:

**1. Heterogeneous universe.** The 24-ETF universe spans asset classes with fundamentally different return drivers — global equities, government bonds, inflation-linked bonds, commodities, high-yield credit, and REITs. Cross-sectional ranking conflates these groups: a commodity ETF in the top quintile driven by a supply shock is not economically comparable to a tech-heavy equity ETF driven by earnings revisions. Ranking across heterogeneous assets reduces the economic signal-to-noise ratio of the momentum factor.

**2. Regime-dependent momentum.** The optimal lookback horizon demonstrably shifted around 2020, likely driven by structural breaks in autocorrelation patterns (zero-rate era, COVID-19 shock, rate normalisation cycle). A fixed parameter, even when selected via walk-forward on recent training data, cannot adapt dynamically to regime changes within the test year. This is consistent with findings in Barroso & Santa-Clara (2015) and Daniel & Moskowitz (2016) showing that momentum is particularly sensitive to volatility regimes.

**3. Reduced strategy expressivity.** With only 18–24 tickers surviving the liquidity filter, a 10%-percentile threshold produces 2–3 long positions and 2–3 short positions. At this scale, idiosyncratic risk dominates and the cross-sectional diversification that underlies the theory of momentum — averaging out noise across many assets — does not hold.

### Academic Context

These results are consistent with the broader empirical literature on momentum deterioration. Stambaugh, Yu & Yuan (2012) and Geczy & Samonov (2017) document a significant decline in momentum profits in diversified multi-asset universes after 2010, attributed to increased arbitrage capacity, risk-off crowding effects, and the structural break caused by the 2008 financial crisis and subsequent unconventional monetary policy. Finding a negative OOS Sharpe is an expected outcome for a simple momentum implementation in this setting.

---

## Architecture

| Module | Role |
|---|---|
| `trade_algorithm.py` | Downloads OHLCV data from Yahoo Finance; applies liquidity filters (min $500k daily volume, $200M AUM); saves `adjclose_wide`, `prices_long`, and `volume_wide` CSVs |
| `loaders/price_loader.py` | `PriceDownloadConfig` dataclass; `download_prices_adjclose`, `quality_checks_prices`, `save_prices_panel` — pivots AdjClose and Volume into wide format |
| `feature_engineering.py` | Computes log returns (5 horizons), rolling volatility (3 windows), rolling drawdown, 200-day MA trend flag, cross-sectional percentile ranks |
| `signal_generation.py` | Loads latest prices → features → CS momentum signals → vol-scaling → normalised weights; writes `signals_cs` and `weights_cs` CSVs |
| `backtester.py` | Historical performance with lagged weights; square-root market-impact cost model (`10 × √participation_rate` bps) with 5 bps flat fallback |
| `robust_evaluation.py` | Grid search over (ranking_horizon, top_pct, vol_window); same market-impact cost model; outputs `robust_results` CSV |
| `walk_forward.py` | Rolling 5-year train / 1-year test windows; parameter selection on train only; concatenated OOS return series; honest out-of-sample reporting |
| `ml_strategy.py` | Logistic regression on per-(date, ticker) feature rows; strict temporal train/test split; CS signal ranking from predicted probabilities |
| `daily_execution.py` | Orchestrates full daily pipeline; risk controls (drawdown circuit breaker, vol targeting, rebalance band, cash budget); submits market orders via Alpaca SDK |
| `dashboard_server.py` | Flask API server (port 5050); endpoints for equity, weights, robustness, logs |
| `templates/dashboard.html` | Single-page dark-theme UI; Chart.js equity curve and positions chart; robustness table with Sharpe heat-map; execution log cards; auto-refresh 60s |
| `setup_scheduler.ps1` | Configures Windows Task Scheduler to run `run_daily.bat` at 16:30 local time under `SYSTEM` account |

---

## Installation

```bash
git clone https://github.com/Rehyvax/TradeAlgorithm.git
cd TradeAlgorithm
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Alpaca Paper Trading credentials:

```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

---

## Usage

```bash
# 1. Download and validate price data
python trade_algorithm.py

# 2. Generate signals and weights
python signal_generation.py

# 3. Backtest with market-impact costs
python backtester.py

# 4. Parameter robustness evaluation
python robust_evaluation.py

# 5. Walk-forward validation (honest OOS estimate)
python walk_forward.py

# 6. ML strategy extension
python ml_strategy.py

# 7. Daily paper trading execution (dry_run=True by default)
python daily_execution.py

# 8. Launch the monitoring dashboard
python dashboard_server.py
# → open http://localhost:5050
```

**Scheduled execution (Windows):** run `setup_scheduler.ps1` as Administrator to register a Task Scheduler job that fires `run_daily.bat` every day at 16:30 local time.

---

## Dashboard

A Flask web dashboard is available at `http://localhost:5050` after running `dashboard_server.py`.

```
┌────────────────────────────────────────────────────────┐
│  TradeAlgorithm            27 Feb 2026 · $100,000      │
├─────────────┬──────────────────────────────────────────┤
│ Equity Curve│  [Normalised equity curve — Chart.js]    │
│ Positions   │  Total Return   Peak Equity   Drawdown   │
│ Robustness  ├────────────────────────────────────────  │
│ Logs        │  [Positions bar chart — green/red]       │
│ About       ├────────────────────────────────────────  │
│             │  [Robustness table — Sharpe heat-map]    │
└─────────────┴──────────────────────────────────────────┘
```

Sections: **Equity Curve** (normalised equity + total return / peak / drawdown metrics), **Positions** (current vol-scaled weights), **Robustness** (top-10 parameter configs by Sharpe), **Logs** (last 30 execution events with order tags), **About** (strategy explanation and glossary in Spanish).

---

## Limitations

- **Paper trading only.** The system connects exclusively to Alpaca's paper trading environment. Slippage, partial fills, and borrow costs for short positions are not modelled.
- **Small universe.** After liquidity filtering, 18–24 ETFs survive. Cross-sectional strategies require large universes (50–500 assets) to diversify away idiosyncratic noise.
- **Market-impact model is approximate.** The square-root model (`10 × √participation_rate` bps) is a standard industry approximation but has not been calibrated to actual execution data.
- **Single factor.** Pure 6–12 month momentum with no conditioning on macro regime, volatility state, or quality filters.

---

## Future Work

| Priority | Extension |
|---|---|
| ✅ Implemented | Walk-forward validation with parameter selection on train data only |
| ✅ Implemented | Square-root market-impact transaction cost model |
| ✅ Implemented | ML extension (logistic regression, per-asset feature rows, CS signal ranking) |
| ✅ Implemented | **Universe segmentation** — momentum applied independently within equity, fixed income, and alternatives groups; OOS Sharpe improved from −0.11 to +0.31 (see Research Findings) |
| Next | **Integrate segmented strategy into `daily_execution.py`** as the default signal source, replacing `signal_generation.py` |
| Next | Volatility regime conditioning — weight signals by a VIX or realised-vol regime indicator to reduce exposure during momentum crash environments (Daniel & Moskowitz, 2016) |
| Next | Expanding universe — include individual stocks or sector ETFs to increase the number of cross-sectional observations and improve the law-of-large-numbers properties of the factor |

---

## References

- Jegadeesh, N., & Titman, S. (1993). *Returns to buying winners and selling losers.* Journal of Finance, 48(1), 65–91.
- Moreira, A., & Muir, T. (2017). *Volatility-managed portfolios.* Journal of Finance, 72(4), 1611–1644.
- Daniel, K., & Moskowitz, T. (2016). *Momentum crashes.* Journal of Financial Economics, 122(2), 221–247.
- Barroso, P., & Santa-Clara, P. (2015). *Momentum has its moments.* Journal of Financial Economics, 116(1), 111–120.

---

## Disclaimer

This repository is provided for research and educational purposes only. It does not constitute financial advice, investment advice, or a solicitation to buy or sell any financial instrument. Past performance of any strategy documented here is not indicative of future results. All results shown are from paper trading simulations and do not represent actual investment returns.
