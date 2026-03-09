# TradeAlgorithm

End-to-end systematic trading pipeline for quantitative research, backtesting, portfolio construction, and Alpaca paper-trading execution.

---

# Overview

TradeAlgorithm is a Python-based systematic trading project designed to demonstrate the full lifecycle of an algorithmic strategy, from raw market data ingestion to signal generation, risk management, backtesting, and daily paper trading execution.

The project combines quantitative finance, data engineering, and automated execution logic in a single modular pipeline.

The repository is intended as a portfolio project showing how a systematic strategy can be researched, validated, and connected to a live paper-trading environment.

---

# Strategy

The core strategy implemented in this project is a **cross-sectional momentum strategy** applied to a universe of assets.

General workflow:

1. Download historical market data  
2. Compute financial features  
3. Rank assets by momentum  
4. Generate long / short signals  
5. Scale positions by volatility  
6. Construct a normalized portfolio  
7. Execute paper trades through Alpaca  

Additional execution logic includes:

- volatility targeting  
- drawdown protection  
- exposure scaling  
- rebalance thresholds  
- minimum trade sizes  
- cash budget constraints  

---

# Pipeline

```text
Market Data
    ↓
Feature Engineering
    ↓
Signal Generation
    ↓
Portfolio Construction
    ↓
Backtesting & Robustness Evaluation
    ↓
Daily Paper Trading Execution

Main Components

The repository is organized around the key stages of a systematic trading workflow:

Script	Purpose
trade_algorithm.py	Updates and stores market data
feature_engineering.py	Computes financial features from price data
signal_generation.py	Generates momentum signals and portfolio weights
backtester.py	Evaluates historical performance of the strategy
robust_evaluation.py	Tests robustness across parameter configurations
daily_execution.py	Runs the automated daily paper trading workflow
ml_strategy.py	Experimental machine learning extension
test_alpaca_connection.py	Tests connectivity with Alpaca API


TradeAlgorithm
│
├── data/                      # stored market data and outputs
├── loaders/                   # utilities for loading and processing data
├── literature/                # reference papers and research material
│
├── trade_algorithm.py         # main data pipeline
├── feature_engineering.py     # financial feature engineering
├── signal_generation.py       # signal generation logic
├── backtester.py              # backtesting engine
├── robust_evaluation.py       # parameter robustness evaluation
├── daily_execution.py         # daily trading execution pipeline
├── ml_strategy.py             # experimental ML strategy
├── test_alpaca_connection.py  # Alpaca API connectivity test
│
├── run_daily.bat              # Windows script for daily execution
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md

## Installation

Clone the repository:

git clone https://github.com/Rehyvax/TradeAlgorithm.git
cd TradeAlgorithm

Install the required dependencies: pip install -r requirements.txt

# Outputs

Depending on the executed scripts, the system may generate:

- historical price datasets
- engineered feature datasets
- trading signals and portfolio weights
- backtesting results
- robustness evaluation outputs
- daily trading logs
- equity history records

---

# Limitations

Current limitations of the project include:

- paper trading only
- simplified transaction cost assumptions
- no detailed slippage model
- limited production hardening

The repository is intended primarily for **research and demonstration purposes**.

---

# Future Improvements

Potential future extensions include:

- transaction cost and slippage modeling
- walk-forward validation
- automated experiment tracking
- improved portfolio optimization
- dashboard visualization of strategy performance
- expanded machine learning research

---

# Disclaimer

This repository is intended for research and educational purposes only.

It does **not** constitute financial advice, investment advice, or a recommendation to buy or sell any financial instrument.