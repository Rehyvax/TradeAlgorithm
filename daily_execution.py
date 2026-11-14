"""Daily execution script for end‑to‑end strategy workflow.

This script orchestrates the complete pipeline required to run the
cross‑sectional momentum strategy in a paper‑trading environment.  It
performs data ingestion, feature extraction, signal generation and
portfolio rebalancing against a paper trading broker (Alpaca).  The
design follows best practices for systematic investment management:

* **Data update** – call ``trade_algorithm.main`` to download the
  latest prices and save them under ``data/prices``.  Running this
  step daily ensures our features are computed from up‑to‑date data.
* **Signal generation** – call ``signal_generation.main`` to compute
  cross‑sectional momentum ranks, raw long/short signals and
  volatility‑scaled weights.  Signals and weights are written into
  ``data/signals`` with a date stamp.
* **Risk management** – compute the most recent equity history from
  prior runs and evaluate simple risk controls: a maximum drawdown
  limit and a target volatility rule.  If the strategy has suffered a
  large drawdown, trading is suspended for the day.  If the portfolio
  volatility is above a target (e.g. 10% annualised), gross exposure
  is scaled down.  These rules are inspired by research showing
  volatility scaling and drawdown filters can improve momentum
  strategies.
* **Rebalance** – load the latest weights (top percentile long, bottom
  percentile short, scaled by volatility) and translate them into
  dollar notional targets using the current account equity.  Compare
  to current market value positions and generate market orders for the
  difference.  Orders are submitted through Alpaca’s paper trading
  API if ``dry_run=False``.
* **Logging** – record a row in ``data/live/equity_history.csv`` with
  the timestamp, equity and gross exposure used.  Write a full JSON
  log of the day’s actions to ``data/live/logs/daily_execution.jsonl``.

The script can be scheduled to run once per day after market close.
By default it runs in dry‑run mode to avoid sending orders until
validated.  See the ``main`` function at the bottom for
configurable parameters.

Prerequisites
-------------

* Install ``alpaca-py`` (``pip install alpaca-py``) and
  ``python-dotenv`` (``pip install python-dotenv``).
* Set your API keys in a ``.env`` file at the project root or as
  environment variables ``ALPACA_API_KEY`` and ``ALPACA_API_SECRET``.
* Ensure the universe and price ingestion pipelines (``trade_algorithm``)
  are run regularly.

This script does not attempt to forecast market conditions.  The
strategy is based purely on cross‑sectional momentum and volatility
scaling; however, the modular structure allows you to plug in
additional features such as learning‑to‑rank models or sentiment
signals in the future.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import trade_algorithm
import signal_generation
import thematic_sleeve
import trend_strategy
from risk_management import RiskConfig, compute_exposure
from universe_segments import run_segmented_strategy


def project_dirs(base_dir: Path) -> Dict[str, Path]:
    """Return a dictionary of important project directories.

    Parameters
    ----------
    base_dir : Path
        Root directory of the project.

    Returns
    -------
    dict[str, Path]
        Mapping of names to Path objects for data, prices, signals,
        live, and logs directories.
    """
    data_dir = base_dir / "data"
    return {
        "data": data_dir,
        "prices": data_dir / "prices",
        "signals": data_dir / "signals",
        "live": data_dir / "live",
        "logs": data_dir / "live" / "logs",
    }


def latest_csv(folder: Path, pattern: str) -> Path:
    """Return the most recent CSV file in ``folder`` matching ``pattern``.

    Parameters
    ----------
    folder : Path
        Directory in which to search.
    pattern : str
        Glob pattern (e.g. ``"weights_cs_*.csv"``).

    Returns
    -------
    Path
        Path to the latest matching file.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {folder} matching {pattern}")
    return files[-1]


def log_jsonl(log_path: Path, event: dict) -> None:
    """Append a JSON event as a line in a log file.

    Parameters
    ----------
    log_path : Path
        The file to which the log event will be appended.
    event : dict
        A serialisable dictionary of event data.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_latest_target_weights(signals_dir: Path, use_segmented: bool = True) -> pd.Series:
    """Load the most recent weight vector from the signals directory.

    If ``use_segmented=True`` (default), looks for the latest
    ``weights_segmented_*.csv`` produced by ``universe_segments.py``.
    Falls back to ``weights_cs_*.csv`` if no segmented file exists.

    Parameters
    ----------
    signals_dir : Path
        Directory containing weight CSV files.
    use_segmented : bool, optional
        If True, prefer segmented strategy weights over CS weights.

    Returns
    -------
    pandas.Series
        Series of target weights indexed by ticker as strings.
    """
    if use_segmented:
        seg_files = sorted(signals_dir.glob("weights_segmented_*.csv"))
        if seg_files:
            w_path = seg_files[-1]
        else:
            print("[WARNING] No weights_segmented_*.csv found - falling back to weights_cs_*.csv")
            w_path = latest_csv(signals_dir, "weights_cs_*.csv")
    else:
        w_path = latest_csv(signals_dir, "weights_cs_*.csv")
    w = pd.read_csv(w_path, index_col=0, parse_dates=True)
    # Take the last row (most recent date) as today's target
    target = w.iloc[-1].copy()
    target.index = target.index.astype(str)
    target = target.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # Ensure normalisation: sum abs = 1 (if not, normalise)
    abs_sum = float(target.abs().sum())
    if abs_sum > 0 and abs_sum != 1.0:
        target = target / abs_sum
    return target
def enforce_weight_constraints(
    w: pd.Series,
    max_positions: int = 10,
    max_weight_per_asset: float = 0.15,
    long_only: bool = True,
) -> pd.Series:
    w = w.copy().fillna(0.0)

    if long_only:
        w[w < 0] = 0.0

    # Keep only top-N by absolute weight
    if max_positions is not None and max_positions > 0:
        keep = w.abs().sort_values(ascending=False).head(max_positions).index
        w = w.where(w.index.isin(keep), 0.0)

    # Cap per-asset weight
    w = w.clip(lower=-max_weight_per_asset, upper=max_weight_per_asset)

    # Renormalize
    s = float(w.abs().sum())
    if s > 0:
        w = w / s
    return w

def alpaca_client() -> TradingClient:
    """Instantiate an Alpaca TradingClient using environment variables.

    Returns
    -------
    alpaca.trading.client.TradingClient
        Configured trading client for paper trading.
    """
    load_dotenv()
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in the environment or .env file"
        )
    return TradingClient(key, secret, paper=True)


def get_account_equity(trading: TradingClient) -> float:
    """Return the current equity of the trading account."""
    acct = trading.get_account()
    return float(acct.equity)
def get_account_cash(trading: TradingClient) -> float:
    """Return current cash available in the account (USD)."""
    acct = trading.get_account()
    return float(acct.cash)

def get_positions(trading: TradingClient) -> Dict[str, float]:
    """Return the current market value of open positions indexed by symbol."""
    positions = trading.get_all_positions()
    out: Dict[str, float] = {}
    for p in positions:
        out[p.symbol] = float(p.market_value)
    return out


def compute_target_notional(
    equity: float,
    target_w: pd.Series,
    gross_exposure: float = 1.0,
    long_only: bool = False,
) -> Dict[str, float]:
    """Convert target weights into dollar notionals for each symbol.

    Parameters
    ----------
    equity : float
        Current account equity in dollars.
    target_w : pandas.Series
        Target weights indexed by symbol.  Sum of absolute values should
        equal 1.
    gross_exposure : float, optional
        Multiplier for exposure.  gross_exposure=1.0 yields
        approximately equity gross exposure (i.e. long and short
        positions summing to equity).  Setting to <1 reduces
        exposure; >1 increases exposure.
    long_only : bool, optional
        If True, negative weights are set to zero and the remaining
        weights are renormalised to sum to 1.  Use this when the
        strategy cannot short.

    Returns
    -------
    dict[str, float]
        Mapping of symbols to target notional dollars.
    """
    w = target_w.copy()
    if long_only:
        w[w < 0] = 0.0
        s = float(w.sum())
        if s > 0:
            w = w / s
    target_notional = (w * (equity * gross_exposure)).to_dict()
    return target_notional


def generate_order_plan(
    equity: float,
    current_mv: Dict[str, float],
    target_notional: Dict[str, float],
    rebalance_band: float = 0.01,
    min_notional_usd: float = 25.0,
) -> List[dict]:
    """Compute differences between current and target positions and build order instructions.

    Parameters
    ----------
    equity : float
        Current account equity.
    current_mv : dict[str, float]
        Mapping of symbols to current market value of positions.
    target_notional : dict[str, float]
        Mapping of symbols to desired dollar exposure.
    rebalance_band : float, optional
        Fraction of equity below which differences are ignored (to
        reduce turnover).  For example, 0.01 ignores differences
        smaller than 1% of equity.
    min_notional_usd : float, optional
        Minimum notional in USD required to create an order; smaller
        trades are skipped.

    Returns
    -------
    list[dict]
        List of order dicts with ``symbol``, ``notional`` and ``side``.
    """
    orders: List[dict] = []
    symbols = sorted(set(current_mv.keys()) | set(target_notional.keys()))
    for sym in symbols:
        cur = float(current_mv.get(sym, 0.0))
        tgt = float(target_notional.get(sym, 0.0))
        delta = tgt - cur
        # Skip tiny differences
        if abs(delta) < min_notional_usd:
            continue
        # Skip differences within rebalance band
        if abs(delta) / max(equity, 1.0) < rebalance_band:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        notional = abs(delta)
        if side == OrderSide.SELL:
            # 0.5% safety margin to avoid "insufficient qty" errors caused by
            # rounding differences between our position estimate and Alpaca's.
            notional *= 0.995
        orders.append({
            "symbol": sym,
            "notional": notional,
            "side": side,
        })
    return orders

def apply_cash_budget(
    orders: List[dict],
    cash_available: float,
    cash_buffer_usd: float = 100.0,
) -> List[dict]:
    """
    Ensure total BUY notionals do not exceed (cash_available - cash_buffer_usd).
    Sells are kept as-is; buys are scaled down proportionally if needed.
    """
    budget = max(0.0, cash_available - cash_buffer_usd)

    sells = [o for o in orders if str(o["side"]) == str(OrderSide.SELL)]
    buys  = [o for o in orders if str(o["side"]) == str(OrderSide.BUY)]

    total_buy = sum(float(o["notional"]) for o in buys)
    if total_buy <= budget or total_buy == 0:
        # Execute sells first anyway (ordering)
        return sells + buys

    scale = budget / total_buy
    scaled_buys = []
    for o in buys:
        o2 = dict(o)
        o2["notional"] = float(o2["notional"]) * scale
        scaled_buys.append(o2)

    return sells + scaled_buys

def submit_orders(
    trading: TradingClient,
    orders: List[dict],
    dry_run: bool = False,
) -> List[dict]:
    """Submit market orders to Alpaca or return a plan if dry_run.

    Parameters
    ----------
    trading : TradingClient
        Alpaca trading client.
    orders : list[dict]
        Each order must contain ``symbol``, ``notional`` and ``side`` keys.
    dry_run : bool, optional
        If True, no orders are sent; the order plan is returned
        unchanged with ``status`` set to ``DRY_RUN``.  If False,
        orders are submitted and ``order_id`` is recorded.

    Returns
    -------
    list[dict]
        List of results containing the original order spec plus
        additional metadata such as ``status`` or ``id``.
    """
    results: List[dict] = []
    for o in orders:
        sym = o["symbol"]
        # Round notional to two decimal places as required by Alpaca.
        notional = round(float(o["notional"]), 2)
        # Skip any orders that are effectively zero after rounding to avoid API errors.
        if notional < 1.00:
            continue
        side = o["side"]
        req = MarketOrderRequest(
            symbol=sym,
            notional=notional,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        if dry_run:
            results.append({
                "symbol": sym,
                "notional": notional,
                "side": str(side),
                "status": "DRY_RUN",
            })
        else:
            try:
                # Submit the order via Alpaca's API. Rounding ensures the notional value
                # conforms to Alpaca's two decimal place requirement.
                order = trading.submit_order(order_data=req)
                results.append({
                    "symbol": sym,
                    "notional": notional,
                    "side": str(side),
                    "id": str(order.id),
                    "status": "SUBMITTED",
                })
            except Exception as exc:
                if "insufficient qty" in str(exc).lower():
                    print(f"[WARNING] {sym}: insufficient qty - skipping. ({exc})")
                    results.append({
                        "symbol": sym,
                        "notional": notional,
                        "side": str(side),
                        "status": "SKIPPED_INSUFFICIENT_QTY",
                    })
                else:
                    raise
    return results


def compute_drawdown(series: pd.Series) -> float:
    """Compute the maximum drawdown of a cumulative returns series.

    Parameters
    ----------
    series : pandas.Series
        Equity or cumulative return series.

    Returns
    -------
    float
        Drawdown as a negative number representing the largest
        percentage drop from a peak.  If no drawdown, returns 0.
    """
    peaks = series.cummax()
    dd = (series - peaks) / peaks
    return float(dd.min())


def estimate_realized_vol(series: pd.Series, window: int = 63) -> float:
    """Estimate annualised volatility of a return series using a rolling window.

    Parameters
    ----------
    series : pandas.Series
        Daily (or business‑day) returns of the portfolio.
    window : int, optional
        Window length for the rolling standard deviation.  Defaults to 63
        trading days (~3 months).

    Returns
    -------
    float
        Annualised volatility (std * sqrt(252)).  If not enough data,
        returns NaN.
    """
    if len(series) < window:
        return float("nan")
    daily_vol = series.rolling(window).std().iloc[-1]
    if pd.isna(daily_vol):
        return float("nan")
    return float(daily_vol * np.sqrt(252))


def build_trend_targets(
    base_dir: Path,
    dirs: Dict[str, Path],
    thematic_allocation: float = 0.30,
) -> pd.Series:
    """Target weights of the second-generation strategy.

    The account is split in two sleeves:

    * a **core** sleeve (``1 - thematic_allocation`` of the account) that
      follows ``trend_strategy.py`` on the ETF universe, and
    * a **thematic** sleeve (``thematic_allocation``) that follows
      ``thematic_sleeve.py`` on ``data/universe_thematic.csv``.

    Capital that the thematic sleeve does not use (because volatility is
    high or trends are negative) is parked in the cash ETF of the core
    sleeve.  If the thematic prices cannot be downloaded the whole
    account follows the core sleeve.

    Parameters
    ----------
    base_dir : Path
        Root directory of the project.
    dirs : dict[str, Path]
        Output of :func:`project_dirs`.
    thematic_allocation : float, optional
        Fraction of the account assigned to the thematic sleeve.

    Returns
    -------
    pandas.Series
        Target weights indexed by ticker.  They sum to at most 1 and are
        *not* renormalised afterwards: the uninvested part is intended.
    """
    etf_prices = pd.read_csv(
        latest_csv(dirs["prices"], "adjclose_wide_*.csv"), index_col=0, parse_dates=True
    ).dropna(axis=1, how="all").sort_index()
    core = trend_strategy.latest_target_weights(etf_prices)

    sleeve = pd.Series(dtype=float)
    if thematic_allocation > 0:
        try:
            theme_prices = thematic_sleeve.download_thematic_prices(base_dir).ffill()
            sleeve = thematic_sleeve.latest_sleeve_weights(theme_prices)
        except Exception as exc:  # network or data problem: fall back to core only
            print(f"[WARNING] Thematic sleeve unavailable ({exc}). Using the core sleeve only.")
            thematic_allocation = 0.0

    target = core * (1.0 - thematic_allocation)
    if thematic_allocation > 0 and not sleeve.empty:
        target = target.add(sleeve * thematic_allocation, fill_value=0.0)
        unused = thematic_allocation * (1.0 - float(sleeve.sum()))
        cash = trend_strategy.CASH_TICKER
        if cash in target.index and unused > 0:
            target[cash] += unused
    target.index = target.index.astype(str)
    return target[target.abs() > 1e-6]


def append_equity_row(equity_path: Path, row: dict) -> None:
    """Append one observation to ``equity_history.csv``."""
    eq_df = pd.DataFrame([row])
    if equity_path.exists():
        eq_df.to_csv(equity_path, mode="a", header=False, index=False)
    else:
        eq_df.to_csv(equity_path, index=False)


def main(
    dry_run: bool = True,
    gross_exposure: float = 1.0,
    long_only: bool = False,
    max_drawdown: float = -0.1,
    target_vol: float = 0.1,
    rebalance_band: float = 0.02,
    min_trade_usd: float = 25.0,
    use_segmented: bool = True,
    strategy: str = "trend",
    thematic_allocation: float = 0.30,
) -> None:
    """Entry point for the daily execution pipeline.

    Parameters
    ----------
    dry_run : bool, optional
        If True, no orders are sent to the broker; orders are
        reported in the log instead.  Set to False to enable
        submission once you are confident the strategy behaves as
        expected.
    gross_exposure : float, optional
        Base gross exposure multiplier applied to target weights.
    long_only : bool, optional
        Disallow short positions by zeroing negative weights.  Only
        relevant for the legacy strategies; the trend strategy is long
        only by construction.
    max_drawdown : float, optional
        Drawdown from the 252-observation peak (negative number) at which
        exposure is cut to its floor.  Exposure starts to fall at half
        this drawdown and the account is moved to cash at twice this
        drawdown.  Unlike the first version of this script, the rule
        looks at the *current* drawdown, so exposure is restored as the
        account recovers.  Default is -0.10.
    target_vol : float, optional
        Annualised volatility target of the account.  If realised
        volatility over the last 63 observations exceeds this level,
        exposure is scaled down proportionally.  Default is 0.10.
    rebalance_band : float, optional
        Fraction of equity representing a threshold below which
        differences between current and target positions are ignored.
        Default is 0.02 (2%), the value used in the backtest of the
        trend strategy.
    min_trade_usd : float, optional
        Minimum notional amount per trade.  Orders smaller than this
        threshold are skipped.
    use_segmented : bool, optional
        Legacy switch, used only when ``strategy="legacy"``.  If True,
        use ``weights_segmented_*.csv``; otherwise ``weights_cs_*.csv``.
    strategy : str, optional
        ``"trend"`` (default) trades the blended trend strategy plus the
        thematic sleeve.  ``"legacy"`` reproduces the original
        cross-sectional momentum behaviour.
    thematic_allocation : float, optional
        Fraction of the account assigned to the thematic sleeve when
        ``strategy="trend"``.  Zero disables the sleeve.
    """
    base_dir = Path(__file__).resolve().parent
    dirs = project_dirs(base_dir)
    dirs["live"].mkdir(parents=True, exist_ok=True)
    dirs["logs"].mkdir(parents=True, exist_ok=True)
    run_flag = dirs["live"] / f"last_run_{datetime.now(timezone.utc).date().isoformat()}.flag"
    if not dry_run and run_flag.exists():
        print("[SKIP] Already ran today. Exiting to avoid duplicate trading.")
        return
    if not dry_run:
        run_flag.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    # Step A: Update data
    trade_algorithm.main()

    # Step B: Generate signals and weights
    today_str = datetime.now(timezone.utc).date().isoformat()
    if strategy == "legacy":
        signal_generation.main()

        # Step B2: Generate segmented weights and save to data/signals/
        prices_df = pd.read_csv(
            latest_csv(dirs["prices"], "adjclose_wide_*.csv"), index_col=0, parse_dates=True
        ).dropna(axis=1, how="all")
        seg_weights: pd.DataFrame = run_segmented_strategy(prices_df)["weights"]
        seg_path = dirs["signals"] / f"weights_segmented_{today_str}.csv"
        seg_weights.to_csv(seg_path)
        print(f"[B2] Segmented weights saved to {seg_path}")

        # Step C: Load latest weights
        target_w = load_latest_target_weights(dirs["signals"], use_segmented=use_segmented)
        target_w = enforce_weight_constraints(
            target_w,
            max_positions=10,
            max_weight_per_asset=0.15,
            long_only=long_only,
        )
    elif strategy == "trend":
        # Step B/C: Blended trend (core) plus thematic sleeve
        target_w = build_trend_targets(base_dir, dirs, thematic_allocation=thematic_allocation)
        dirs["signals"].mkdir(parents=True, exist_ok=True)
        live_path = dirs["signals"] / f"weights_live_{today_str}.csv"
        target_w.rename("weight").to_csv(live_path, index_label="Ticker")
        print(f"[B] Live target weights saved to {live_path}")
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'trend' or 'legacy'.")

    # Step D: Connect to broker and fetch account info
    trading = alpaca_client()
    equity = get_account_equity(trading)
    positions = get_positions(trading)
    cash = get_account_cash(trading)

    timestamp = datetime.now(timezone.utc).isoformat()

    # Step E: Risk controls based on past equity history
    equity_path = dirs["live"] / "equity_history.csv"
    risk_cfg = RiskConfig(
        target_vol=target_vol,
        dd_soft=max_drawdown / 2.0,
        dd_hard=max_drawdown,
        dd_kill=max_drawdown * 2.0,
    )
    equity_series = pd.Series([equity], dtype=float)
    if equity_path.exists():
        hist = pd.read_csv(equity_path)
        if not hist.empty:
            equity_series = pd.concat([hist["equity"], equity_series], ignore_index=True)
    risk = compute_exposure(equity_series, risk_cfg)
    scaled_exposure = gross_exposure * risk["exposure"]

    if risk["suspect_print"]:
        # The broker reported an equity value that is implausibly far from
        # the previous ones.  Do not trade on it; look again tomorrow.
        print(f"Equity print {equity:,.2f} looks wrong. No orders today.")
        log_jsonl(
            dirs["logs"] / "daily_execution.jsonl",
            {
                "timestamp_utc": timestamp,
                "equity": equity,
                "halt_reason": "suspect_equity_print",
                "gross_exposure": gross_exposure,
            },
        )
        append_equity_row(equity_path, {
            "timestamp_utc": timestamp, "equity": equity,
            "gross_exposure": 0.0, "dry_run": dry_run, "n_orders": 0,
        })
        return

    if risk["kill_switch"]:
        print(
            f"Drawdown reached {risk_cfg.dd_kill:.0%} within the last "
            f"{risk_cfg.kill_cooldown} runs. Moving the account to cash."
        )
        target_w = target_w * 0.0
    elif risk["exposure"] < 1.0:
        print(
            f"Risk overlay: drawdown {risk['drawdown']:.2%} -> x{risk['dd_multiplier']:.2f}, "
            f"volatility -> x{risk['vol_multiplier']:.2f}. Exposure {scaled_exposure:.2f}."
        )

    # Step F: Compute target dollar positions
    if strategy == "legacy":
        target_notional = compute_target_notional(
            equity=equity,
            target_w=target_w,
            gross_exposure=scaled_exposure,
            long_only=long_only,
        )
    else:
        # Trend weights already embed the volatility target and the cash
        # position, so they must not be renormalised to sum to one.
        target_notional = (target_w * (equity * scaled_exposure)).to_dict()

    # Step G: Generate order plan and apply cash budget
    orders = generate_order_plan(
        equity=equity,
        current_mv=positions,
        target_notional=target_notional,
        rebalance_band=rebalance_band,
        min_notional_usd=min_trade_usd,
    )
    orders = apply_cash_budget(orders, cash_available=cash, cash_buffer_usd=max(50.0, 0.01 * equity))

    # Step H: Submit orders
    results = submit_orders(trading, orders, dry_run=dry_run)
    n_orders = len(results)

    # Step I: Record today's equity and exposure
    append_equity_row(equity_path, {
        "timestamp_utc": timestamp, "equity": equity,
        "gross_exposure": scaled_exposure, "dry_run": dry_run, "n_orders": n_orders,
    })

    # Step J: Write audit log
    log_jsonl(
        dirs["logs"] / "daily_execution.jsonl",
        {
            "timestamp_utc": timestamp,
            "equity": equity,
            "dry_run": dry_run,
            "strategy": strategy,
            "gross_exposure": scaled_exposure,
            "drawdown": risk["drawdown"],
            "dd_multiplier": risk["dd_multiplier"],
            "vol_multiplier": risk["vol_multiplier"],
            "kill_switch": risk["kill_switch"],
            "long_only": long_only,
            "orders": results,
        },
    )

    print(f"[OK] daily_execution completed. dry_run={dry_run}, orders={n_orders}")
    print(f"Equity history logged to: {equity_path}")
    print(f"Audit log JSONL:      {dirs['logs'] / 'daily_execution.jsonl'}")


if __name__ == "__main__":
    # Paper trading with the blended trend strategy and the thematic sleeve.
    # Use strategy="legacy" to reproduce the original behaviour.
    main(dry_run=False, gross_exposure=0.98, long_only=True)
