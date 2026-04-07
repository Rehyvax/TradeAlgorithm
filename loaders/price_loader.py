from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import date
import pandas as pd
import yfinance as yf


@dataclass
class PriceDownloadConfig:
    """Configuration object for downloading price data via yfinance.

    Parameters
    ----------
    start : str, optional
        The inclusive start date for the series in ``YYYY-MM-DD`` format. Defaults to
        ``"2005-01-01"``.
    interval : str, optional
        The sampling frequency for the data. Valid values include ``"1d"`` (daily),
        ``"1wk"`` (weekly) and ``"1mo"`` (monthly). Defaults to daily.
    auto_adjust : bool, optional
        If set to ``True`` yfinance will return adjusted close prices in the
        ``Close`` column. In our pipeline we set this to ``False`` and explicitly
        rename the ``Adj Close`` column to ``AdjClose`` for clarity. Defaults to
        ``False``.
    progress : bool, optional
        Whether yfinance should display a progress bar while downloading. Defaults to
        ``False``.
    """

    start: str = "2005-01-01"
    interval: str = "1d"
    auto_adjust: bool = False
    progress: bool = False


def download_prices_adjclose(
    tickers: list[str],
    cfg: PriceDownloadConfig,
) -> pd.DataFrame:
    """Download adjusted OHLCV price data for a list of tickers.

    For each ticker this function requests historical price data from yfinance.
    Retrieving each series individually avoids complications with multi-index
    DataFrames that arise when downloading multiple tickers simultaneously.
    The adjusted close column is renamed to ``AdjClose`` to explicitly
    differentiate it from the raw ``Close`` price. If a given column is
    missing in the downloaded data (for example, some instruments may not
    report volume), a new column filled with ``pandas.NA`` is created to
    ensure a consistent schema. The resulting rows are concatenated into a
    single long-format DataFrame with columns ``Date``, ``Ticker``, ``AdjClose``,
    ``Close`` and ``Volume``.

    Parameters
    ----------
    tickers : list[str]
        A list of ticker symbols to download.
    cfg : PriceDownloadConfig
        Configuration specifying start date, interval and other yfinance
        options.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with one row per date per ticker containing
        adjusted close, close and volume data. If no data is available for
        any ticker, an empty DataFrame with the expected columns is returned.
    """
    frames: list[pd.DataFrame] = []
    for tkr in tickers:
        # Request data for a single ticker. Downloading individually yields a
        # standard DataFrame rather than a MultiIndex when multiple tickers are
        # requested at once. This avoids indexing issues with certain asset
        # classes.
        data = yf.download(
            tickers=tkr,
            start=cfg.start,
            interval=cfg.interval,
            auto_adjust=cfg.auto_adjust,
            progress=cfg.progress,
        )

        # Skip tickers with no returned data
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(tkr, axis=1, level=1, drop_level=True)

        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        # Rename the adjusted close column to a consistent name
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "AdjClose"})
        # If an adjusted close column is missing, fall back to the raw close column
        if "AdjClose" not in df.columns and "Close" in df.columns:
            df["AdjClose"] = df["Close"]
        # Add a column identifying the ticker
        df["Ticker"] = tkr
        # Ensure required columns exist; create them with missing values if needed
        for col in ["AdjClose", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = pd.NA
        # Reset the index to expose the date as a column and select the schema
        frame = df.reset_index()[["Date", "Ticker", "AdjClose", "Close", "Volume"]]
        frames.append(frame)

    # If no data was collected, return an empty DataFrame with the expected columns
    if not frames:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Close", "Volume"])

    # Concatenate the per-ticker frames into a single DataFrame
    return pd.concat(frames, ignore_index=True)


def quality_checks_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply basic data quality flags to a price DataFrame.

    For convenience the loader marks rows that may require manual
    inspection before using the data in backtests or live models. The
    following flags are added as boolean columns to the returned DataFrame:

    * ``flag_missing_adjclose`` – ``True`` when ``AdjClose`` is missing
      (``NA``) or non‑positive. Adjusted close should always be present and
      strictly positive; zeros or negative values often indicate corrupted
      data.
    * ``flag_bad_volume`` – ``True`` when ``Volume`` is missing or
      negative. Negative volumes are nonsensical and missing values may
      require imputation or exclusion.
    * ``flag_duplicate`` – ``True`` when duplicate ``(Date, Ticker)`` rows
      are detected. Duplicate rows could result from overlapping downloads
      or accidental concatenations.

    Parameters
    ----------
    prices : pandas.DataFrame
        A long-format DataFrame returned by :func:`download_prices_adjclose`
        containing ``Date``, ``Ticker``, ``AdjClose``, ``Close`` and
        ``Volume`` columns.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with three additional boolean flag
        columns. The original data columns are preserved.
    """
    p = prices.copy()
    # Flag rows where the adjusted close is missing or non-positive
    p["flag_missing_adjclose"] = p["AdjClose"].isna() | (p["AdjClose"] <= 0)
    # Flag rows with missing or negative volume
    p["flag_bad_volume"] = p["Volume"].isna() | (p["Volume"] < 0)
    # Identify duplicate rows by (Date, Ticker)
    dup = p.duplicated(subset=["Date", "Ticker"], keep=False)
    p["flag_duplicate"] = dup
    return p


def save_prices_panel(prices: pd.DataFrame, base_dir: Path) -> Path:
    """Persist price data to disk in both long and wide formats.

    The long format contains one row per date per ticker and is suitable
    for most analytics, while the wide format pivots the data to produce
    a matrix with dates on the index and tickers on the columns, storing
    adjusted close prices. The files are written into a ``data/prices``
    subdirectory relative to ``base_dir`` and include a date stamp in
    their filenames for versioning. The long-format file path is returned.

    Parameters
    ----------
    prices : pandas.DataFrame
        A cleaned long-format price DataFrame with columns ``Date``,
        ``Ticker`` and ``AdjClose`` among others.
    base_dir : pathlib.Path
        The root directory of your project. Output files will be stored in
        ``base_dir / 'data' / 'prices'``.

    Returns
    -------
    pathlib.Path
        The path of the long-format CSV file that was written. The wide-format
        file is also written but its path is not returned.
    """
    prices_dir = base_dir / "data" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    # Use today's date as a version stamp on the output filenames
    stamp = date.today().isoformat()
    long_path = prices_dir / f"prices_long_{stamp}.csv"
    # Write the long-format data
    prices.to_csv(long_path, index=False)

    # Pivot the adjusted close into a wide matrix for backtesting convenience
    wide = prices.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
    wide_path = prices_dir / f"adjclose_wide_{stamp}.csv"
    wide.to_csv(wide_path)

    # Pivot daily volume into the same wide format for market-impact cost models
    vol_wide = prices.pivot(index="Date", columns="Ticker", values="Volume").sort_index()
    vol_wide_path = prices_dir / f"volume_wide_{stamp}.csv"
    vol_wide.to_csv(vol_wide_path)

    return long_path
