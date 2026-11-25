"""
Replication of the Ranked Asset Allocation Model (Giordano, 2018) and its
comparison with the blended trend strategy.

Source
------
Giordano, G. (2020). Ranked Asset Allocation Model. Journal of Technical
Analysis, 70, 106 to 125. Winner of the 2018 Charles H. Dow Award.

The paper ranks eleven ETFs of the 7Twelve portfolio every month on four
factors, holds the five best at 20% each, and sends to cash any of the five
with negative four-month momentum.  It reports a Sharpe ratio of 1.94 from
July 2004 to November 2017, gross of costs.

What this script does
---------------------
1. Rebuilds the model with the rules printed in the paper.
2. Runs it on a proxy universe with the same mix of asset classes, because
   eight of the twelve original ETFs are not in this repository's data.
3. Splits the test into the paper's own period (2005 to 2017) and the years
   after publication (2018 onwards), which are a genuine out-of-sample test
   of the published rules.
4. Measures each factor's contribution by removing them one at a time.
5. Tests the paper's ideas grafted onto the trend strategy.

Choices where the paper is silent or ambiguous
----------------------------------------------
* Factor weights and the tie breaker ``x`` are not disclosed.  We use
  momentum 1, volatility 0.5, correlation 0.5 and a negligible momentum
  tie breaker.
* Rank 1 is the best asset and the five lowest total ranks are held, as the
  selection rule and the minus sign on the trend term imply.
* The repository holds closes only, so the ATR uses absolute close to close
  changes and the breakout bands use closes.  The lower band, printed as
  "ATR plus the highest low of 105 periods", is read as the lowest close of
  105 periods minus the ATR.
* Weights decided at the last close of a month earn returns from the next
  day, and every trade pays a one-way cost.

Usage::

    python raam_replication.py
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from performance_metrics import compute_performance_metrics, deflated_sharpe_ratio
from trend_strategy import TrendConfig, run_trend_strategy

CASH = "SHY"

# 7Twelve risky ETFs and the closest asset available in this repository.
PROXY_UNIVERSE: Dict[str, str] = {
    "VV (US large cap)": "IWF",
    "IJH (US mid cap)": "IWD",
    "IJR (US small cap)": "IWN",
    "RWR (US real estate)": "IWO",
    "EFA (developed ex US)": "EFA",
    "EEM (emerging markets)": "EEM",
    "DBC (commodities)": "USO",
    "VAW (materials)": "GLD",
    "AGG (US aggregate bonds)": "LQD",
    "TIP (inflation linked)": "TIP",
    "IGOV (international treasuries)": "BWX",
}


@dataclass(frozen=True)
class RaamConfig:
    """Parameters of the Ranked Asset Allocation Model.

    Defaults follow the paper; weights of the ranks are our assumption.
    """

    momentum_window: int = 84
    correlation_window: int = 84
    ewma_lambda: float = 0.94
    vol_smoothing: int = 10
    atr_window: int = 42
    upper_window: int = 63
    lower_window: int = 105
    w_momentum: float = 1.0
    w_volatility: float = 0.5
    w_correlation: float = 0.5
    use_trend: bool = True
    n_hold: int = 5
    absolute_filter: bool = True
    cost_bps: float = 5.0


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

def ewma_volatility(prices: pd.DataFrame, lam: float, smoothing: int) -> pd.DataFrame:
    """RiskMetrics volatility of daily log returns, smoothed over ``smoothing`` days."""
    r = np.log(prices).diff()
    var = (r ** 2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var * 252.0).rolling(smoothing).mean()


def average_correlation(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean pairwise correlation of each asset with the others over ``window`` days."""
    r = prices.pct_change()
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    corr = r.rolling(window, min_periods=window).corr()
    for t in r.index[window:]:
        c = corr.loc[t].to_numpy().copy()
        np.fill_diagonal(c, np.nan)
        valid = ~np.isnan(c).all(axis=1)
        if valid.sum() < 2:
            continue
        out.loc[t, r.columns[valid]] = np.nanmean(c[valid][:, valid], axis=1)
    return out


def atr_breakout_state(prices: pd.DataFrame, cfg: RaamConfig) -> pd.DataFrame:
    """Persistent breakout state: +2 after an upside breakout, -2 after a downside one."""
    atr = prices.diff().abs().rolling(cfg.atr_window).mean()
    upper = prices.rolling(cfg.upper_window).max().shift(1) + atr.shift(1)
    lower = prices.rolling(cfg.lower_window).min().shift(1) - atr.shift(1)
    signal = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    signal[prices > upper] = 2.0
    signal[prices < lower] = -2.0
    return signal.ffill().fillna(0.0)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def raam_weights(prices: pd.DataFrame, cfg: RaamConfig, risky: List[str]) -> pd.DataFrame:
    """Monthly target weights of the model, forward filled to every day."""
    px = prices[risky]
    mom = px / px.shift(cfg.momentum_window) - 1.0
    vol = ewma_volatility(px, cfg.ewma_lambda, cfg.vol_smoothing)
    cor = average_correlation(px, cfg.correlation_window)
    trend = atr_breakout_state(px, cfg)

    cols = risky + [CASH]
    rows = {}
    for t in month_ends(prices.index):
        ok = mom.loc[t].notna() & vol.loc[t].notna() & cor.loc[t].notna()
        if ok.sum() < cfg.n_hold:
            continue
        m, v, c, tr = mom.loc[t][ok], vol.loc[t][ok], cor.loc[t][ok], trend.loc[t][ok]
        total = (
            cfg.w_momentum * m.rank(ascending=False)
            + cfg.w_volatility * v.rank(ascending=True)
            + cfg.w_correlation * c.rank(ascending=True)
            - (tr if cfg.use_trend else 0.0)
            - m * 1e-6
        )
        chosen = total.nsmallest(cfg.n_hold).index
        w = pd.Series(0.0, index=cols)
        for a in chosen:
            if cfg.absolute_filter and m[a] <= 0:
                w[CASH] += 1.0 / cfg.n_hold
            else:
                w[a] += 1.0 / cfg.n_hold
        rows[t] = w
    target = pd.DataFrame(rows).T
    return target.reindex(prices.index).ffill().fillna(0.0)


def latest_raam_weights(
    prices: pd.DataFrame,
    cfg: RaamConfig = RaamConfig(),
    risky: Optional[List[str]] = None,
    history: int = 400,
) -> pd.Series:
    """Current target weights of the model, for live execution.

    Only the last ``history`` rows are used, which is enough for every
    factor window and keeps the correlation step fast.
    """
    risky = risky or list(PROXY_UNIVERSE.values())
    px = prices[risky + [CASH]].ffill().iloc[-history:]
    w = raam_weights(px, cfg, risky).iloc[-1]
    return w[w > 0]


def backtest(weights: pd.DataFrame, prices: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Daily returns of ``weights`` decided at close ``t`` and held over ``t + 1``.

    Costs are charged on the traded notional when the target changes.
    """
    rets = prices[weights.columns].pct_change().fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover.shift(1).fillna(0.0) * cost_bps / 1e4


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(ret: pd.Series, rf: pd.Series, start: str, end: Optional[str] = None) -> Dict[str, float]:
    """Key metrics of ``ret`` between ``start`` and ``end``."""
    r = ret.loc[start:end]
    m = compute_performance_metrics(r, rf.loc[r.index])
    return {
        "Sharpe": m["sharpe_ratio"],
        "CAGR": m["annualised_return"],
        "Vol": m["annualised_volatility"],
        "MaxDD": m["max_drawdown"],
    }


def table(results: Dict[str, pd.Series], rf: pd.Series, periods: Dict[str, tuple]) -> pd.DataFrame:
    """One row per strategy, one block of columns per period."""
    out = {}
    for name, ret in results.items():
        row = {}
        for label, (a, b) in periods.items():
            for k, v in summarise(ret, rf, a, b).items():
                row[(label, k)] = v
        out[name] = row
    return pd.DataFrame(out).T


def main() -> None:
    files = sorted(glob.glob(os.path.join("data", "prices", "adjclose_wide_*.csv")))
    prices = pd.read_csv(files[-1], index_col=0, parse_dates=True).sort_index().ffill()
    risky = list(PROXY_UNIVERSE.values())
    rf = prices[CASH].pct_change().fillna(0.0)

    periods = {
        "Paper period 2006-2017": ("2006-01-01", "2017-11-30"),
        "After publication 2018-2026": ("2018-01-01", None),
        "Full 2006-2026": ("2006-01-01", None),
    }

    base = RaamConfig()
    variants: Dict[str, RaamConfig] = {
        "RAAM gross (as in paper)": replace(base, cost_bps=0.0),
        "RAAM net 5 bps": base,
        "RAAM without trend": replace(base, use_trend=False),
        "RAAM without correlation": replace(base, w_correlation=0.0),
        "RAAM without volatility": replace(base, w_volatility=0.0),
        "RAAM momentum only": replace(base, w_volatility=0.0, w_correlation=0.0, use_trend=False),
        "RAAM without absolute filter": replace(base, absolute_filter=False),
        "RAAM hold 3": replace(base, n_hold=3),
        "RAAM hold 7": replace(base, n_hold=7),
    }

    results: Dict[str, pd.Series] = {}
    for name, cfg in variants.items():
        w = raam_weights(prices, cfg, risky)
        results[name] = backtest(w, prices, cfg.cost_bps)

    # The repository's own trend strategy on the same proxy universe and on its full universe.
    trend_full = run_trend_strategy(prices)
    results["Trend strategy (full universe)"] = trend_full["returns"]
    eq = prices[risky].pct_change().fillna(0.0).mean(axis=1)
    results["Equal weight 11 proxies"] = eq

    tab = table(results, rf, periods)
    pd.set_option("display.width", 220)
    print("Proxy universe:")
    for k, v in PROXY_UNIVERSE.items():
        print(f"  {k:<34} -> {v}")
    print()
    print(tab.to_string(float_format=lambda v: f"{v:.2f}"))

    # Selection-adjusted evidence for the RAAM variants over the full period.
    full = {k: v.loc["2006-01-01":] for k, v in results.items() if k.startswith("RAAM") and "gross" not in k}
    sharpes = pd.Series({k: summarise(v, rf, "2006-01-01")["Sharpe"] for k, v in full.items()})
    dsr = deflated_sharpe_ratio(full["RAAM net 5 bps"], sharpes, rf.loc[full["RAAM net 5 bps"].index])
    print(f"\nDeflated Sharpe Ratio of RAAM net 5 bps across {len(sharpes)} variants: {dsr:.3f}")

    out = os.path.join("data", "research")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(results).loc["2006-01-01":].to_csv(os.path.join(out, "raam_returns.csv"))
    tab.to_csv(os.path.join(out, "raam_comparison.csv"))
    print(f"Saved returns and table to {out}")


if __name__ == "__main__":
    main()
