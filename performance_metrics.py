"""Shared performance and statistical-significance metrics.

The original modules (``legacy/backtester.py``, ``legacy/robust_evaluation.py`` and
``legacy/ml_strategy.py``) each carry their own copy of a metrics function.  Those
copies divide the geometric annualised return by volatility and assume a
zero risk-free rate, which overstates the Sharpe ratio whenever cash yields
are material (2023 onwards).  This module is the single reference
implementation used by the second-generation strategy modules
(``trend_strategy.py``, ``thematic_sleeve.py`` and
``short_horizon_study.py``).

Besides the usual statistics it implements two tools that answer the
question "is this Sharpe ratio distinguishable from luck?":

* **Probabilistic Sharpe Ratio (PSR)**: the probability that the true
  Sharpe ratio exceeds a benchmark given the sample length, skewness and
  kurtosis of the returns (Bailey & Lopez de Prado, 2012).
* **Deflated Sharpe Ratio (DSR)**: the PSR evaluated against the Sharpe
  ratio one would expect from the best of ``n_trials`` unskilled
  strategies (Bailey & Lopez de Prado, 2014).  Every parameter grid in
  this repository is a multiple-testing exercise, so the best
  configuration must be judged against this higher bar.

All functions take *simple* daily returns.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def _excess(returns: pd.Series, risk_free: Optional[pd.Series | float]) -> pd.Series:
    """Subtract a risk-free series (or constant daily rate) from returns."""
    r = returns.dropna().astype(float)
    if risk_free is None:
        return r
    if isinstance(risk_free, pd.Series):
        return r - risk_free.reindex(r.index).fillna(0.0)
    return r - float(risk_free)


def compute_performance_metrics(
    returns: pd.Series,
    risk_free: Optional[pd.Series | float] = None,
) -> Dict[str, float]:
    """Compute summary statistics for a series of daily simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Daily simple returns of the strategy, net of costs.
    risk_free : pandas.Series or float, optional
        Daily risk-free return.  Either a series aligned with ``returns``
        (for example the daily return of a short-duration Treasury ETF) or
        a constant daily rate.  ``None`` means zero.

    Returns
    -------
    dict
        ``cumulative_return``, ``annualised_return`` (geometric),
        ``annualised_volatility``, ``sharpe_ratio`` (arithmetic mean of
        excess returns over their standard deviation, annualised),
        ``sortino_ratio``, ``max_drawdown``, ``calmar_ratio`` and
        ``n_days``.
    """
    r = returns.dropna().astype(float)
    keys = [
        "cumulative_return", "annualised_return", "annualised_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "n_days",
    ]
    if len(r) < 2:
        return {k: np.nan for k in keys}
    ex = _excess(r, risk_free)
    curve = (1.0 + r).cumprod()
    cumulative = float(curve.iloc[-1] - 1.0)
    ann_return = float(curve.iloc[-1] ** (TRADING_DAYS / len(r)) - 1.0)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ex_std = float(ex.std())
    sharpe = float(ex.mean() / ex_std * np.sqrt(TRADING_DAYS)) if ex_std > 0 else np.nan
    downside = float(np.sqrt((ex.clip(upper=0.0) ** 2).mean()))
    sortino = float(ex.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan
    max_dd = float((curve / curve.cummax() - 1.0).min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "cumulative_return": cumulative,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "n_days": float(len(r)),
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    risk_free: Optional[pd.Series | float] = None,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``.

    Parameters
    ----------
    returns : pandas.Series
        Daily simple returns.
    benchmark_sharpe : float, optional
        Annualised Sharpe ratio to beat.  Defaults to zero.
    risk_free : pandas.Series or float, optional
        Daily risk-free return, as in :func:`compute_performance_metrics`.

    Returns
    -------
    float
        A probability in [0, 1].  Values above 0.95 are conventionally
        read as significant.
    """
    ex = _excess(returns, risk_free)
    n = len(ex)
    if n < 30 or ex.std() == 0:
        return float("nan")
    sr = ex.mean() / ex.std()                      # per-period Sharpe
    sr_star = benchmark_sharpe / np.sqrt(TRADING_DAYS)
    skew = float(sps.skew(ex))
    kurt = float(sps.kurtosis(ex, fisher=False))
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr_star) * np.sqrt(n - 1.0) / denom
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected maximum annualised Sharpe among ``n_trials`` unskilled strategies.

    Parameters
    ----------
    n_trials : int
        Number of configurations that were tried.
    sharpe_std : float
        Standard deviation of the annualised Sharpe ratios across trials.

    Returns
    -------
    float
        The annualised Sharpe ratio that pure selection would be expected
        to produce.
    """
    if n_trials < 2 or not np.isfinite(sharpe_std) or sharpe_std <= 0:
        return 0.0
    a = sps.norm.ppf(1.0 - 1.0 / n_trials)
    b = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: Iterable[float],
    risk_free: Optional[pd.Series | float] = None,
) -> float:
    """Deflated Sharpe Ratio of the selected strategy.

    Parameters
    ----------
    returns : pandas.Series
        Daily simple returns of the configuration that was selected.
    trial_sharpes : iterable of float
        Annualised Sharpe ratios of *all* configurations that were tried,
        including the selected one.
    risk_free : pandas.Series or float, optional
        Daily risk-free return.

    Returns
    -------
    float
        Probability that the selected strategy has a positive true Sharpe
        after accounting for the number of trials.
    """
    trials = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    hurdle = expected_max_sharpe(len(trials), float(trials.std(ddof=1)) if len(trials) > 1 else 0.0)
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe=hurdle, risk_free=risk_free)


def bootstrap_sharpe_interval(
    returns: pd.Series,
    risk_free: Optional[pd.Series | float] = None,
    block: int = 21,
    n_boot: int = 1000,
    level: float = 0.90,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval for the Sharpe ratio from a block bootstrap.

    Blocks of ``block`` consecutive days are resampled with replacement so
    that volatility clustering is preserved.

    Parameters
    ----------
    returns : pandas.Series
        Daily simple returns.
    risk_free : pandas.Series or float, optional
        Daily risk-free return.
    block : int, optional
        Block length in days.
    n_boot : int, optional
        Number of bootstrap samples.
    level : float, optional
        Coverage of the interval.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    (float, float)
        Lower and upper bound of the annualised Sharpe ratio.
    """
    ex = _excess(returns, risk_free).to_numpy()
    n = len(ex)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([ex[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        out[b] = sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    lo, hi = np.nanquantile(out, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
