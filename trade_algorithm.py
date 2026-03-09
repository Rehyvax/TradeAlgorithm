"""Main script for downloading ETF metadata and price data.

This module orchestrates the data ingestion pipeline for a multi‑asset
trading system. It loads a list of ETF tickers from a static universe CSV,
retrieves accompanying metadata (assets under management and average volume)
from the Yahoo Finance API, and downloads historical price series using
utility functions defined in ``price_loader.py``. Both metadata and price
data are saved to disk under the ``data/`` directory with date stamps
embedded in the filenames. Comments and messages are written in English to
ensure clarity and professionalism.

The pipeline is composed of three major steps:

1. **Load the universe** – read a CSV file (``data/universe.csv``) that
   specifies which tickers to fetch. The file must contain at least a
   ``Ticker`` column.
2. **Download metadata** – for each ticker, query Yahoo Finance to obtain
   the current total assets under management (``totalAssets``) and the
   average daily trading volume (``averageVolume``). The resulting metadata
   DataFrame is written to ``data/metadata/etf_metadata_YYYY‑MM‑DD.csv``.
3. **Download price history** – fetch adjusted OHLCV data for each ticker
   individually via yfinance. The data is checked for missing or anomalous
   values before being saved in both long and wide formats under
   ``data/prices/``.

Running this script repeatedly will create new timestamped files, which
facilitates versioning and auditing of the data used in research and
backtesting. Adjust the configuration (start date, interval) as needed by
editing the ``PriceDownloadConfig`` in ``price_loader.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

# Import the price loader utilities. These live in ``price_loader.py`` in the
# same directory as this script. If you restructure your project into
# packages, adjust this import accordingly.
# Import the price loader utilities from the local module.  The original
# version expected these functions to live in a ``loaders`` package, but this
# repository organises them at the top level in ``price_loader.py``.  If you
# restructure your project into packages, adjust this import accordingly.
from loaders.price_loader import (
    PriceDownloadConfig,
    download_prices_adjclose,
    quality_checks_prices,
    save_prices_panel,
)


def update_etf_metadata(tickers: List[str]) -> pd.DataFrame:
    """Retrieve ETF metadata using Yahoo Finance.

    For each ticker this helper instantiates a :class:`yfinance.Ticker` object
    and extracts key descriptive statistics from its ``info`` dictionary. The
    fields collected are total assets under management (AUM) and average
    trading volume, both expressed in US dollars. A timestamp is added so
    that the age of the metadata is explicit.

    Parameters
    ----------
    tickers : list[str]
        List of ETF ticker symbols to query.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``Ticker``, ``Date``, ``AUM_USD`` and
        ``AvgVolume``. If a field is unavailable for a given ticker it will
        contain ``None``.
    """
    rows: list[dict] = []
    today = date.today()
    for tkr in tickers:
        t = yf.Ticker(tkr)
        info = t.info or {}
        rows.append(
            {
                "Ticker": tkr,
                "Date": today,
                # Some tickers may not have totalAssets; default to None
                "AUM_USD": info.get("totalAssets"),
                "AvgVolume": info.get("averageVolume"),
            }
        )
    return pd.DataFrame(rows)
def filter_universe_by_liquidity(
        universe: pd.DataFrame,
        metadata: pd.DataFrame,
        min_avg_volume: float = 500_000,
        min_aum_usd: float = 200_000_000,
    ) -> list[str]:
        md = metadata.copy()
        md["AvgVolume"] = pd.to_numeric(md["AvgVolume"], errors="coerce")
        md["AUM_USD"] = pd.to_numeric(md["AUM_USD"], errors="coerce")

        merged = universe.merge(md[["Ticker", "AvgVolume", "AUM_USD"]], on="Ticker", how="left")

        keep = merged[
            (merged["AvgVolume"].fillna(0) >= min_avg_volume)
            & (merged["AUM_USD"].fillna(0) >= min_aum_usd)
        ]["Ticker"].astype(str).tolist()
        return keep
def main() -> None:
    """Execute the full data ingestion pipeline.

    This function performs the following operations:

    * Reads the universe CSV from ``data/universe.csv``.
    * Retrieves and writes ETF metadata via :func:`update_etf_metadata`.
    * Downloads historical adjusted price data using
      :func:`download_prices_adjclose` with default configuration.
    * Applies quality flags to identify missing or anomalous price rows.
    * Saves both long‑format and wide‑format price panels to disk.

    The function prints informative messages about the location of the
    generated files to aid debugging and auditing. Errors from yfinance are
    allowed to propagate; catch exceptions at a higher level if you need
    finer control over error handling.
    """
    # Determine the base directory as the directory containing this script
    base_dir = Path(__file__).resolve().parent
    universe_path = base_dir / "data" / "universe.csv"
    
    # Step 1: Load the universe. Require at least a ``Ticker`` column.
    universe = pd.read_csv(universe_path)
    if "Ticker" not in universe.columns:
        raise KeyError(
            "Universe CSV must contain a 'Ticker' column. Found columns: "
            f"{list(universe.columns)}"
        )
    tickers = (
        universe["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if not tickers:
        raise ValueError("Universe is empty – no tickers found to download.")
    metadata_dir = base_dir / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Step 2: Fetch and save metadata
    metadata = update_etf_metadata(tickers)
    tickers = filter_universe_by_liquidity(universe, metadata, min_avg_volume=500_000, min_aum_usd=200_000_000)
    raw_universe_path = metadata_dir / f"universe_raw_{date.today()}.csv"
    pd.DataFrame({"Ticker": tickers}).to_csv(raw_universe_path, index=False)
    print(f"Saved raw universe to {raw_universe_path}")
    if not tickers:
        raise ValueError("After liquidity/AUM filter, universe is empty.")
    filtered_universe_path = metadata_dir / f"universe_filtered_{date.today()}.csv"
    pd.DataFrame({"Ticker": tickers}).to_csv(filtered_universe_path, index=False)
    print(f"Saved filtered universe to {filtered_universe_path}")
    metadata_filtered = metadata[metadata["Ticker"].isin(tickers)].copy()
    metadata_file = metadata_dir / f"etf_metadata_{date.today()}.csv"
    metadata_filtered.to_csv(metadata_file, index=False)
    print(f"Saved metadata to {metadata_file}")
    
    # Step 3: Download historical prices using default configuration
    cfg = PriceDownloadConfig()
    prices = download_prices_adjclose(tickers, cfg)
    
    # Step 4: Apply quality flags
    qc_prices = quality_checks_prices(prices)
    issues = qc_prices[
        qc_prices["flag_missing_adjclose"]
        | qc_prices["flag_bad_volume"]
        | qc_prices["flag_duplicate"]
    ]
    if not issues.empty:
        issues_path = base_dir / "data" / "prices" / "issues_prices.csv"
        issues_path.parent.mkdir(parents=True, exist_ok=True)
        issues.to_csv(issues_path, index=False)
        print(
            f"Warning: found issues in downloaded prices. See {issues_path} for details."
        )
    
    # Step 5: Save the cleaned price data (drop flags) in both formats
    prices_clean = qc_prices.drop(
        columns=["flag_missing_adjclose", "flag_bad_volume", "flag_duplicate"]
    )
    prices_path = save_prices_panel(prices_clean, base_dir)
    print(f"Saved price panel to {prices_path}")
    

    

if __name__ == "__main__":
    main()