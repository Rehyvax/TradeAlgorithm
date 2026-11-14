"""
trend_strategy.py
=================

Second-generation core strategy: blended time-series trend with risk
budgeting, volatility targeting and a staggered weekly rebalance.

Why the original strategy was replaced
--------------------------------------
The first strategy in this repository ranks all ETFs against each other,
buys the top quantile and shorts the bottom quantile.  Walk-forward
validation gave it an out-of-sample Sharpe ratio of -0.11, and applying
the ranking inside each asset class (``universe_segments.py``) lifted it
to +0.31.  Three design choices explain most of the shortfall:

1. **The short leg.**  With 18 to 24 broad ETFs the "losers" are whole
   asset classes.  Shorting them means paying their risk premium, and a
   long-short book on so few assets is dominated by noise.
2. **A single lookback chosen by in-sample Sharpe.**  The best lookback
   jumped between 126 and 252 days from window to window.  Selecting the
   maximum of twelve noisy numbers is selection bias, not adaptation.
3. **Daily re-ranking.**  Signals were recomputed and traded every day
   although the information in a 6 to 12 month return barely changes
   overnight.

What this module does instead
-----------------------------
* **Time-series trend, long or flat.**  Each asset is judged against its
  own history: it is held when its return over the lookback is positive
  and replaced by cash otherwise (Moskowitz, Ooi & Pedersen, 2012).
* **A blend of speeds, no selection.**  The trend score is the average
  vote of three lookbacks (3, 6 and 12 months).  Blending slower and
  faster signals handles turning points better than either extreme
  (Goulding, Harvey & Mazzoleni, 2023) and removes the need to pick a
  winner.  A one-month lookback was tested and left out: it raised
  turnover by half without improving the result.
* **Relative strength inside each sleeve.**  Within an asset class the
  assets with the strongest volatility-adjusted trend receive a larger
  share.  This keeps the finding of ``universe_segments.py``, that
  ranking only makes sense among comparable assets, as a tilt rather
  than as an all-or-nothing rule.
* **Risk budgeting.**  Assets are weighted by inverse volatility inside
  a sleeve and sleeves are scaled to contribute similar risk, so that
  low-volatility bond ETFs do not crowd out everything else.
* **Volatility targeting without leverage.**  The book is scaled to an
  annualised volatility target using the trailing covariance matrix and
  capped at 100 % gross exposure.  What is not invested sits in the cash
  ETF.
* **Staggered weekly rebalance.**  The book is the average of five
  sub-portfolios, each refreshed on a different day of the week.  A
  single weekly rebalance made the Sharpe ratio move between 0.43 and
  0.54 depending only on the weekday chosen; staggering removes that
  luck (Hoffstein, Faber & Braun, 2020).  Trades smaller than 2 % of the
  book are skipped.

The parameters are conventional values from the literature and were not
optimised.  ``main`` reports a robustness table over neighbouring
choices together with the Deflated Sharpe Ratio so that the reader can
judge how much of the result depends on the exact configuration.

Usage::

    python trend_strategy.py

Outputs (written to ``data/signals/``):
  - ``weights_trend_YYYY-MM-DD.csv``        daily target weights
  - ``trend_returns_YYYY-MM-DD.csv``        daily net returns
  - ``trend_robustness_YYYY-MM-DD.csv``     one row per tested configuration

References
----------
* Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time series
  momentum. Journal of Financial Economics, 104(2), 228-250.
* Goulding, C., Harvey, C., & Mazzoleni, M. (2023). Momentum turning
  points. Journal of Financial Economics, 149(3), 378-406.
* Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. Journal
  of Finance, 72(4), 1611-1644.
* Hoffstein, C., Faber, N., & Braun, S. (2020). Rebalance timing luck:
  the (dumb) luck of smart beta. SSRN working paper 3673910.
* Bailey, D., & Lopez de Prado, M. (2014). The deflated Sharpe ratio.
  Journal of Portfolio Management, 40(5), 94-107.
"""

from __future__ import annotations

import datetime as dt
import glob
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from performance_metrics import (
    bootstrap_sharpe_interval,
    compute_performance_metrics,
    deflated_sharpe_ratio,
)

# Sleeves of comparable assets.  The cash ETF is deliberately left out: it
# is the destination of whatever the strategy does not invest.
SLEEVES: Dict[str, List[str]] = {
    "equity":       ["EEM", "EFA", "IWD", "IWF", "IWN", "IWO", "SCZ"],
    "fixed_income": ["BWX", "EMB", "HYG", "LQD", "TIP", "TLT"],
    "alternatives": ["GLD", "USO", "BIZD", "BKLN", "RWR", "DBA", "DBB"],
}
CASH_TICKER = "SHY"


@dataclass(frozen=True)
class TrendConfig:
    """Configuration of the blended trend strategy.

    Parameters
    ----------
    lookbacks : tuple of int
        Lookback windows, in trading days, whose signs are averaged into
        the trend score.
    vol_window : int
        Window for asset volatilities and for the covariance matrix.
    target_vol : float
        Annualised volatility target of the invested book.
    rebalance_every : int
        Number of trading days between rebalances.
    rebalance_offset : int
        Shifts the rebalance schedule when ``tranched`` is False; used to
        measure how much a result depends on the day of the week.
    tranched : bool
        If True, hold the average of ``rebalance_every`` staggered
        sub-portfolios instead of a single one.
    no_trade_band : float
        Minimum change in an asset's weight that triggers a trade.
    relative_tilt : float
        Strength of the within-sleeve relative-strength tilt.  With 0.5
        the strongest asset of a sleeve receives 1.5 times its base weight
        and the weakest 0.5 times.  Zero disables the tilt.
    reversal_tilt : float
        Strength of the five-day reversal tilt (see
        ``short_horizon_study.py``).  Zero by default: at a weekly
        rebalance it adds turnover without improving the result.
    long_only : bool
        If True, assets in a downtrend are replaced by cash.  If False
        they are shorted.
    risk_budget : bool
        If True, sleeves are scaled to similar ex-ante volatility.
    max_weight : float
        Cap on the absolute weight of a single asset.
    max_gross : float
        Cap on gross exposure (1.0 means no leverage).
    cost_bps : float
        One-way transaction cost in basis points of traded notional.
    warmup : int
        Number of initial rows without positions.
    """

    lookbacks: Tuple[int, ...] = (63, 126, 252)
    vol_window: int = 63
    target_vol: float = 0.10
    rebalance_every: int = 5
    rebalance_offset: int = 0
    tranched: bool = True
    no_trade_band: float = 0.02
    relative_tilt: float = 0.5
    reversal_tilt: float = 0.0
    long_only: bool = True
    risk_budget: bool = True
    max_weight: float = 0.30
    max_gross: float = 1.0
    cost_bps: float = 5.0
    warmup: int = 300


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def compute_trend_score(
    prices: pd.DataFrame,
    lookbacks: Tuple[int, ...] = (63, 126, 252),
    long_only: bool = True,
) -> pd.DataFrame:
    """Average the sign of the return over several lookbacks.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices (dates x tickers).
    lookbacks : tuple of int, optional
        Lookback windows in trading days.
    long_only : bool, optional
        If True the score is clipped at zero, so it reads as the fraction
        of lookbacks that point up.

    Returns
    -------
    pandas.DataFrame
        Scores in [0, 1] (long only) or [-1, 1].  Rows where the longest
        lookback is not yet available are NaN.
    """
    votes = sum(np.sign(prices / prices.shift(lb) - 1.0) for lb in lookbacks) / len(lookbacks)
    votes = votes.where(prices.shift(max(lookbacks)).notna())
    return votes.clip(lower=0.0) if long_only else votes


def compute_relative_strength(
    prices: pd.DataFrame,
    vol: pd.DataFrame,
    lookbacks: Tuple[int, ...],
    sleeves: Dict[str, List[str]],
) -> pd.DataFrame:
    """Percentile rank of the volatility-adjusted trend inside each sleeve.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices.
    vol : pandas.DataFrame
        Annualised volatility of each asset.
    lookbacks : tuple of int
        Lookbacks whose volatility-adjusted returns are averaged.
    sleeves : dict
        Mapping of sleeve name to tickers.

    Returns
    -------
    pandas.DataFrame
        Ranks in (0, 1]; NaN where the asset has no history or belongs to
        no sleeve.
    """
    eligible = prices.shift(max(lookbacks)).notna()
    z = sum(
        (prices / prices.shift(lb) - 1.0) / (vol * np.sqrt(lb / 252.0)) for lb in lookbacks
    ) / len(lookbacks)
    rank = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for tickers in sleeves.values():
        cols = [t for t in tickers if t in prices.columns]
        if cols:
            rank[cols] = z[cols].where(eligible[cols]).rank(axis=1, pct=True)
    return rank


def compute_reversal_zscore(prices: pd.DataFrame, vol: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Return over ``window`` days in units of its expected standard deviation."""
    return (prices / prices.shift(window) - 1.0) / (vol * np.sqrt(window / 252.0))


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def _prepare(prices: pd.DataFrame, cfg: TrendConfig, sleeves: Dict[str, List[str]]) -> dict:
    """Pre-compute, as NumPy arrays, every panel needed to build weights."""
    returns = prices.pct_change()
    vol = returns.rolling(cfg.vol_window).std() * np.sqrt(252.0)
    eligible = prices.shift(max(cfg.lookbacks)).notna()
    conviction = compute_trend_score(prices, cfg.lookbacks, cfg.long_only)
    if cfg.relative_tilt:
        rank = compute_relative_strength(prices, vol, cfg.lookbacks, sleeves)
        conviction = conviction * (1.0 + cfg.relative_tilt * (2.0 * rank - 1.0))
    if cfg.reversal_tilt:
        z = compute_reversal_zscore(prices, vol)
        conviction = conviction * (1.0 - cfg.reversal_tilt * z.clip(-1.0, 1.0))
    inv_vol = (1.0 / vol).where(eligible).replace([np.inf, -np.inf], np.nan)
    signal = (conviction * inv_vol).fillna(0.0)
    columns = list(prices.columns)
    return {
        "returns": returns.fillna(0.0).to_numpy(),
        "inv_vol": inv_vol.fillna(0.0).to_numpy(),
        "signal": signal.to_numpy(),
        "sleeve_idx": [
            np.array([columns.index(t) for t in tickers if t in columns], dtype=int)
            for tickers in sleeves.values()
        ],
    }


def _weights_on(i: int, panels: dict, cfg: TrendConfig) -> np.ndarray:
    """Target weights of the risky assets using information up to row ``i``."""
    window = panels["returns"][i - cfg.vol_window + 1: i + 1]
    cov = np.cov(window, rowvar=False) * 252.0
    target = np.zeros(window.shape[1])
    for idx in panels["sleeve_idx"]:
        inv_vol = panels["inv_vol"][i, idx]
        total = inv_vol.sum()
        if idx.size == 0 or total <= 0:
            continue
        fully_invested = inv_vol / total            # sleeve if every asset trended up
        sleeve = panels["signal"][i, idx] / total
        if cfg.risk_budget:
            sleeve_cov = cov[np.ix_(idx, idx)]
            sleeve = sleeve / np.sqrt(max(fully_invested @ sleeve_cov @ fully_invested, 1e-12))
        target[idx] = sleeve
    gross = np.abs(target).sum()
    if gross <= 0:
        return target
    target = target / gross
    book_vol = np.sqrt(max(target @ cov @ target, 1e-12))
    target = target * min(cfg.max_gross, cfg.target_vol / book_vol)
    return np.clip(target, -cfg.max_weight, cfg.max_weight)


def compute_daily_targets(
    prices: pd.DataFrame,
    cfg: TrendConfig = TrendConfig(),
    sleeves: Optional[Dict[str, List[str]]] = None,
    first_row: Optional[int] = None,
) -> pd.DataFrame:
    """Target weights of the risky assets recomputed on every date.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices, sorted by date.
    cfg : TrendConfig, optional
        Strategy configuration.
    sleeves : dict, optional
        Mapping of sleeve name to tickers.  Defaults to :data:`SLEEVES`.
    first_row : int, optional
        First row to compute.  Earlier rows are zero.  Defaults to
        ``cfg.warmup``.

    Returns
    -------
    pandas.DataFrame
        Untranched targets (dates x tickers).  The cash ticker is zero
        here; it is filled by :func:`build_target_weights`.
    """
    sleeves = sleeves or SLEEVES
    panels = _prepare(prices, cfg, sleeves)
    first_row = cfg.warmup if first_row is None else max(first_row, cfg.vol_window)
    targets = np.zeros(prices.shape)
    for i in range(first_row, len(prices)):
        targets[i] = _weights_on(i, panels, cfg)
    return pd.DataFrame(targets, index=prices.index, columns=prices.columns)


def apply_no_trade_band(weights: pd.DataFrame, band: float) -> pd.DataFrame:
    """Keep the previous weight of an asset while its target moves less than ``band``.

    Parameters
    ----------
    weights : pandas.DataFrame
        Desired weights (dates x tickers).
    band : float
        Minimum absolute change in weight that triggers a trade.

    Returns
    -------
    pandas.DataFrame
        Weights actually held.
    """
    if band <= 0:
        return weights
    desired = weights.to_numpy()
    held = np.empty_like(desired)
    current = desired[0].copy()
    for i in range(len(desired)):
        move = np.abs(desired[i] - current) >= band
        current = np.where(move, desired[i], current)
        held[i] = current
    return pd.DataFrame(held, index=weights.index, columns=weights.columns)


def build_target_weights(
    prices: pd.DataFrame,
    cfg: TrendConfig = TrendConfig(),
    sleeves: Optional[Dict[str, List[str]]] = None,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """Daily weights of the strategy, including the cash ETF.

    With ``cfg.tranched=True`` the book is the average of
    ``rebalance_every`` sub-portfolios, each rebalanced once per period on
    a different day.  That average equals the rolling mean of the daily
    targets over the last ``rebalance_every`` days.  With
    ``cfg.tranched=False`` a single portfolio is rebalanced on the dates
    given by ``rebalance_every`` and ``rebalance_offset``.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices (dates x tickers), sorted by date.
    cfg : TrendConfig, optional
        Strategy configuration.
    sleeves : dict, optional
        Mapping of sleeve name to tickers.  Defaults to :data:`SLEEVES`.
    cash_ticker : str, optional
        Ticker that receives the uninvested fraction.  If it is not in
        ``prices`` the remainder is left as cash earning nothing.

    Returns
    -------
    pandas.DataFrame
        Weights (dates x tickers).
    """
    prices = prices.sort_index()
    daily = compute_daily_targets(prices, cfg, sleeves)
    if cfg.tranched:
        weights = daily.rolling(cfg.rebalance_every, min_periods=1).mean()
    else:
        steps = np.arange(len(prices)) - cfg.warmup - cfg.rebalance_offset
        is_rebalance = pd.Series((steps >= 0) & (steps % cfg.rebalance_every == 0), index=prices.index)
        weights = daily.where(is_rebalance, np.nan).ffill().fillna(0.0)
    weights = apply_no_trade_band(weights, cfg.no_trade_band)
    return _fill_cash(weights, cfg, cash_ticker)


def _fill_cash(weights: pd.DataFrame, cfg: TrendConfig, cash_ticker: str) -> pd.DataFrame:
    """Assign the uninvested fraction to the cash ticker (long-only books)."""
    if cash_ticker in weights.columns and cfg.long_only:
        weights = weights.copy()
        risky = weights.drop(columns=[cash_ticker]).sum(axis=1)
        weights[cash_ticker] = (cfg.max_gross - risky).clip(lower=0.0)
        weights.iloc[: cfg.warmup, weights.columns.get_loc(cash_ticker)] = 0.0
    return weights


def latest_target_weights(
    prices: pd.DataFrame,
    cfg: TrendConfig = TrendConfig(),
    sleeves: Optional[Dict[str, List[str]]] = None,
    cash_ticker: str = CASH_TICKER,
) -> pd.Series:
    """Weights the strategy wants to hold after the last row of ``prices``.

    Used by ``daily_execution.py``.  Only the last ``rebalance_every``
    rows are computed.  The no-trade band is not applied here because the
    execution engine has its own rebalance band on live positions.
    """
    prices = prices.sort_index()
    n_rows = cfg.rebalance_every if cfg.tranched else 1
    daily = compute_daily_targets(prices, cfg, sleeves, first_row=len(prices) - n_rows)
    weights = daily.iloc[-n_rows:].mean().to_frame().T
    live_cfg = replace(cfg, warmup=0)
    return _fill_cash(weights, live_cfg, cash_ticker).iloc[0]


def run_trend_strategy(
    prices: pd.DataFrame,
    cfg: TrendConfig = TrendConfig(),
    sleeves: Optional[Dict[str, List[str]]] = None,
    cash_ticker: str = CASH_TICKER,
) -> dict:
    """Backtest the blended trend strategy.

    Weights decided with the close of day ``t`` earn the return of day
    ``t + 1``.  Costs are charged on the absolute change in weights.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices (dates x tickers).
    cfg : TrendConfig, optional
        Strategy configuration.
    sleeves : dict, optional
        Mapping of sleeve name to tickers.
    cash_ticker : str, optional
        Ticker used as cash and as risk-free proxy for the Sharpe ratio.

    Returns
    -------
    dict
        ``metrics`` (see ``performance_metrics``), ``weights``,
        ``returns`` (daily, net of costs), ``turnover`` and ``risk_free``.
    """
    prices = prices.sort_index().dropna(axis=1, how="all")
    weights = build_target_weights(prices, cfg, sleeves, cash_ticker)
    asset_returns = prices.pct_change().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    gross = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
    net = gross - turnover * cfg.cost_bps / 10_000.0
    risk_free = asset_returns[cash_ticker] if cash_ticker in asset_returns.columns else None
    live = net.iloc[cfg.warmup + 1:]
    metrics = compute_performance_metrics(live, risk_free)
    metrics["avg_gross_exposure"] = float(weights.drop(columns=[cash_ticker], errors="ignore").abs().sum(axis=1).iloc[cfg.warmup:].mean())
    metrics["annual_turnover"] = float(turnover.iloc[cfg.warmup:].mean() * 252.0)
    return {"metrics": metrics, "weights": weights, "returns": net, "turnover": turnover, "risk_free": risk_free}


# ---------------------------------------------------------------------------
# Research helpers
# ---------------------------------------------------------------------------

ROBUSTNESS_GRID: Dict[str, dict] = {
    "baseline":                 {},
    "lookbacks 21/63/126/252":  {"lookbacks": (21, 63, 126, 252)},
    "lookbacks 126/252":        {"lookbacks": (126, 252)},
    "lookback 252 only":        {"lookbacks": (252,)},
    "lookbacks 21/63":          {"lookbacks": (21, 63)},
    "vol window 21":            {"vol_window": 21},
    "vol window 126":           {"vol_window": 126},
    "target vol 8%":            {"target_vol": 0.08},
    "target vol 12%":           {"target_vol": 0.12},
    "rebalance every 10 days":  {"rebalance_every": 10},
    "rebalance every 21 days":  {"rebalance_every": 21},
    "no trade band 0%":         {"no_trade_band": 0.0},
    "no trade band 1%":         {"no_trade_band": 0.01},
    "no trade band 3%":         {"no_trade_band": 0.03},
    "no relative tilt":         {"relative_tilt": 0.0},
    "relative tilt 1.0":        {"relative_tilt": 1.0},
    "reversal tilt 0.5":        {"reversal_tilt": 0.5},
    "no risk budget":           {"risk_budget": False},
    "long-short":               {"long_only": False},
    "cost 10 bps":              {"cost_bps": 10.0},
}


def robustness_table(prices: pd.DataFrame, start: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Evaluate the neighbouring configurations in :data:`ROBUSTNESS_GRID`.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide adjusted-close prices.
    start : str
        First date of the evaluation period (``YYYY-MM-DD``).

    Returns
    -------
    (pandas.DataFrame, dict)
        One row of metrics per configuration, and the daily returns of
        each configuration keyed by name.
    """
    rows, series = [], {}
    for name, overrides in ROBUSTNESS_GRID.items():
        res = run_trend_strategy(prices, replace(TrendConfig(), **overrides))
        ret = res["returns"].loc[start:]
        m = compute_performance_metrics(ret, res["risk_free"])
        rows.append({
            "configuration": name,
            "sharpe_ratio": m["sharpe_ratio"],
            "annualised_return": m["annualised_return"],
            "annualised_volatility": m["annualised_volatility"],
            "max_drawdown": m["max_drawdown"],
            "annual_turnover": float(res["turnover"].loc[start:].mean() * 252.0),
        })
        series[name] = ret
    return pd.DataFrame(rows), series


def benchmark_returns(prices: pd.DataFrame, cash_ticker: str = CASH_TICKER) -> Dict[str, pd.Series]:
    """Passive benchmarks built from the same universe.

    * ``equal_weight``: all risky ETFs, rebalanced daily.
    * ``60_40``: 30 % IWF, 30 % IWD, 40 % TLT when those tickers exist.
    """
    r = prices.pct_change()
    risky = [c for c in prices.columns if c != cash_ticker]
    out = {"equal_weight": r[risky].mean(axis=1)}
    if {"IWF", "IWD", "TLT"}.issubset(prices.columns):
        out["60_40"] = 0.3 * r["IWF"] + 0.3 * r["IWD"] + 0.4 * r["TLT"]
    return out


def main() -> None:
    price_dir = os.path.join("data", "prices")
    files = sorted(glob.glob(os.path.join(price_dir, "adjclose_wide_*.csv")))
    if not files:
        raise FileNotFoundError(f"No price files found in {price_dir}. Run trade_algorithm.py first.")
    prices = pd.read_csv(files[-1], index_col=0, parse_dates=True).dropna(axis=1, how="all").sort_index()
    print(f"Loaded: {len(prices)} rows x {len(prices.columns)} tickers")
    print(f"Range : {prices.index[0].date()} -- {prices.index[-1].date()}")

    # Evaluate from the first date on which the original walk-forward had
    # out-of-sample results (five years after the start of the data).
    start = str((prices.index[0] + pd.DateOffset(years=5)).date())
    result = run_trend_strategy(prices)
    rf = result["risk_free"]
    oos = result["returns"].loc[start:]
    metrics = compute_performance_metrics(oos, rf)

    print()
    print(f"Blended trend strategy, {start} to {prices.index[-1].date()} (net of {TrendConfig().cost_bps:.0f} bps)")
    for k in ["sharpe_ratio", "annualised_return", "annualised_volatility", "max_drawdown", "calmar_ratio"]:
        print(f"  {k}: {metrics[k]:.4f}")
    lo, hi = bootstrap_sharpe_interval(oos, rf)
    print(f"  sharpe 90% interval (block bootstrap): [{lo:.2f}, {hi:.2f}]")

    print()
    print("Benchmarks over the same period")
    for name, ret in benchmark_returns(prices).items():
        m = compute_performance_metrics(ret.loc[start:], rf)
        print(f"  {name:<13} Sharpe {m['sharpe_ratio']:+.2f}  return {m['annualised_return']:+.1%}  "
              f"vol {m['annualised_volatility']:.1%}  max DD {m['max_drawdown']:.1%}")

    print()
    print("Calendar-year Sharpe ratio")
    by_year = oos.groupby(oos.index.year)
    for year, chunk in by_year:
        m = compute_performance_metrics(chunk, rf)
        print(f"  {year}: Sharpe {m['sharpe_ratio']:+.2f}  return {m['annualised_return']:+.1%}")

    print()
    print("Robustness to neighbouring configurations")
    table, series = robustness_table(prices, start)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    dsr = deflated_sharpe_ratio(series["baseline"], table["sharpe_ratio"], rf)
    print(f"\n  Sharpe range across {len(table)} configurations: "
          f"{table['sharpe_ratio'].min():.2f} to {table['sharpe_ratio'].max():.2f}")
    print(f"  Deflated Sharpe Ratio of the baseline: {dsr:.3f}")

    signals_dir = os.path.join("data", "signals")
    os.makedirs(signals_dir, exist_ok=True)
    today_str = dt.date.today().isoformat()
    result["weights"].to_csv(os.path.join(signals_dir, f"weights_trend_{today_str}.csv"))
    result["returns"].to_csv(os.path.join(signals_dir, f"trend_returns_{today_str}.csv"), header=["return"])
    table.to_csv(os.path.join(signals_dir, f"trend_robustness_{today_str}.csv"), index=False)
    print(f"\nSaved weights, returns and robustness table to {signals_dir}")


if __name__ == "__main__":
    main()
