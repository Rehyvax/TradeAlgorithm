"""
make_figures.py
---------------
Figures used in README.md.  Each figure is rebuilt from the outputs of
the research scripts, so the images always match the numbers printed by
those scripts.

* ``screenshots/trend_equity.png``: growth of one dollar and drawdown of
  the core sleeve against the two passive benchmarks.
* ``screenshots/trend_robustness.png``: Sharpe ratio of the twenty
  neighbouring configurations of the core sleeve.
* ``screenshots/thematic_sleeve.png``: thematic sleeve against the
  buy-and-hold basket.  Needs ``thematic_returns_*.csv`` (run
  ``thematic_sleeve.py`` first).
* ``screenshots/short_horizon.png``: Sharpe ratio of the k-day momentum
  rule on SMH and SOXX as a function of k.  Needs
  ``short_horizon_rules_*.csv`` (run ``short_horizon_study.py`` first).

A figure whose input is missing is skipped with a message.

Usage
-----
    python make_figures.py
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trend_strategy

# Palette: fixed order, the core strategy always takes the first colour.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

OUT_DIR = "screenshots"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
    })


def _latest(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def _growth_and_drawdown(series: dict, colours: dict, title: str, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    for name, r in series.items():
        wealth = (1.0 + r.fillna(0.0)).cumprod()
        ax1.plot(wealth.index, wealth.values, color=colours[name], linewidth=2, label=name)
        ax1.annotate(
            name, xy=(wealth.index[-1], wealth.iloc[-1]), xytext=(6, 0),
            textcoords="offset points", va="center", color=INK, fontsize=9,
        )
        dd = _drawdown(r)
        ax2.plot(dd.index, dd.values * 100, color=colours[name], linewidth=1.5)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_title(title)
    ax1.set_ylabel("Growth of 1 (log scale)")
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Drawdown (%)")
    ax2.axhline(0, color=GRID, linewidth=0.8)
    fig.align_ylabels((ax1, ax2))
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def figure_trend_equity(prices: pd.DataFrame) -> None:
    start = str((prices.index[0] + pd.DateOffset(years=5)).date())
    result = trend_strategy.run_trend_strategy(prices)
    bench = trend_strategy.benchmark_returns(prices)
    series = {
        "Blended trend": result["returns"].loc[start:],
        "Equal weight": bench["equal_weight"].loc[start:],
    }
    colours = {"Blended trend": BLUE, "Equal weight": ORANGE}
    if "60_40" in bench:
        series["60/40"] = bench["60_40"].loc[start:]
        colours["60/40"] = AQUA
    _growth_and_drawdown(
        series, colours,
        "Core sleeve against passive portfolios of the same ETFs (net of 5 bps)",
        os.path.join(OUT_DIR, "trend_equity.png"),
    )


def figure_trend_robustness(prices: pd.DataFrame) -> None:
    path = _latest(os.path.join("data", "signals", "trend_robustness_*.csv"))
    if path is None:
        start = str((prices.index[0] + pd.DateOffset(years=5)).date())
        table, _ = trend_strategy.robustness_table(prices, start)
    else:
        table = pd.read_csv(path)
    table = table.sort_values("sharpe_ratio")
    colours = [BLUE if c == "baseline" else "#9ec5f4" for c in table["configuration"]]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.axvline(0.55, color=ORANGE, linewidth=2, linestyle="--", zorder=1)
    ax.barh(table["configuration"], table["sharpe_ratio"], color=colours, height=0.7, zorder=2)
    for y, v in enumerate(table["sharpe_ratio"]):
        ax.text(v - 0.008, y, f"{v:.2f}", va="center", ha="right",
                color="white" if table["configuration"].iloc[y] == "baseline" else INK, fontsize=8,
                zorder=3, bbox={"facecolor": colours[y], "edgecolor": "none", "pad": 1.5})
    ax.text(0.553, len(table) - 0.4, "Equal weight 0.55", color=INK, fontsize=9, va="bottom")
    ax.set_xlim(0, max(0.7, table["sharpe_ratio"].max() + 0.08))
    ax.set_xlabel("Sharpe ratio over SHY")
    ax.set_title("Core sleeve: Sharpe ratio across 20 neighbouring configurations")
    ax.grid(axis="y", visible=False)
    out = os.path.join(OUT_DIR, "trend_robustness.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def figure_thematic() -> None:
    path = _latest(os.path.join("data", "signals", "thematic_returns_*.csv"))
    if path is None:
        print("Skipped thematic_sleeve.png: run thematic_sleeve.py first.")
        return
    r = pd.read_csv(path, index_col=0, parse_dates=True)
    r = r.loc["2006-01-01":]
    _growth_and_drawdown(
        {"Thematic sleeve": r["sleeve"], "Basket, buy and hold": r["basket"]},
        {"Thematic sleeve": BLUE, "Basket, buy and hold": ORANGE},
        "Thematic sleeve against the equal-weighted basket (19 stocks, net of 5 bps)",
        os.path.join(OUT_DIR, "thematic_sleeve.png"),
    )


def figure_short_horizon() -> None:
    path = _latest(os.path.join("data", "research", "short_horizon_rules_*.csv"))
    if path is None:
        print("Skipped short_horizon.png: run short_horizon_study.py first.")
        return
    rules = pd.read_csv(path)
    rules["period_start"] = rules["period_start"].astype(str).str[:10]
    period = sorted(rules["period_start"].unique())[0]
    sub = rules[rules["period_start"] == period]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for ticker, colour in (("SMH", BLUE), ("SOXX", ORANGE)):
        t = sub[sub["ticker"] == ticker].set_index("rule")["sharpe_ratio"]
        mom = t[t.index.str.startswith("momentum_")]
        k = mom.index.str.extract(r"momentum_(\d+)d")[0].astype(int).values
        order = np.argsort(k)
        x = np.arange(len(k))
        ax.plot(x, mom.values[order], color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{ticker}, momentum rule")
        if "buy_and_hold" in t.index:
            ax.axhline(t["buy_and_hold"], color=colour, linewidth=1.2, linestyle="--",
                       label=f"{ticker}, buy and hold")
        labels = [f"{v}" for v in k[order]]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.axvspan(-0.4, 3.4, color=GRID, alpha=0.5, zorder=0)
    ax.text(1.5, ax.get_ylim()[0] + 0.04, "1 to 5 days", ha="center", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("Lookback k (trading days): hold when the k-day return is positive")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(f"Short-window momentum in semiconductors, since {period[:4]}")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    out = os.path.join(OUT_DIR, "short_horizon.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    _style()
    os.makedirs(OUT_DIR, exist_ok=True)
    price_path = _latest(os.path.join("data", "prices", "adjclose_wide_*.csv"))
    if price_path is None:
        raise FileNotFoundError("No price files in data/prices. Run trade_algorithm.py first.")
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True).dropna(axis=1, how="all").sort_index()
    figure_trend_equity(prices)
    figure_trend_robustness(prices)
    figure_thematic()
    figure_short_horizon()


if __name__ == "__main__":
    main()
