"""Quick dashboard to visualise live equity history.

This script reads the equity history produced by ``daily_execution.py``
and plots the normalised equity curve over time.  It uses matplotlib
for simplicity.  In a production setting you might replace this
with a web‑based dashboard (e.g. Plotly Dash, Bokeh or a Jupyter
notebook), but this lightweight version suffices for inspection and
debugging.

Usage::

    python live_dashboard.py

Ensure that ``daily_execution.py`` has been run at least once and
that ``data/live/equity_history.csv`` exists.  The chart shows
equity normalised to 1 at the start of the series.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    equity_path = base_dir / "data" / "live" / "equity_history.csv"
    if not equity_path.exists():
        raise FileNotFoundError(
            f"Equity history not found at {equity_path}. Run daily_execution.py first."
        )
    df = pd.read_csv(equity_path)
    if df.empty:
        raise ValueError("Equity history file is empty.")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc")
    # Normalise equity series to start at 1
    df["equity_norm"] = df["equity"] / df["equity"].iloc[0]
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp_utc"], df["equity_norm"], label="Portfolio equity (norm)")
    plt.title("Equity Curve from Paper Trading")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Normalised Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()