"""
dashboard_server.py
===================

Flask web server that exposes the trading pipeline data via a JSON API and
serves the single-page dashboard UI.

Run with:
    python dashboard_server.py

Requires: flask, flask-cors
    pip install flask flask-cors
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd
from flask import Flask, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))


def _latest(pattern: str) -> str:
    """Return the most recent file matching *pattern*, or raise FileNotFoundError."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return files[-1]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/equity")
def equity():
    try:
        path = os.path.join(BASE, "data", "live", "equity_history.csv")
        df = pd.read_csv(path)
        first = df["equity"].iloc[0]
        records = [
            {
                "timestamp": row["timestamp_utc"],
                "equity": round(float(row["equity"]), 2),
                "equity_norm": round(float(row["equity"]) / first, 6) if first else 1.0,
            }
            for _, row in df.iterrows()
        ]
        return jsonify(records)
    except FileNotFoundError:
        return jsonify({"error": "no data yet"})


@app.route("/api/weights")
def weights():
    try:
        path = _latest(os.path.join(BASE, "data", "signals", "weights_cs_*.csv"))
        df = pd.read_csv(path, index_col=0)
        last_date = str(df.index[-1])
        last_row = df.iloc[-1]
        weights_list = [
            {"ticker": ticker, "weight": round(float(w), 6)}
            for ticker, w in last_row.items()
            if pd.notna(w) and w != 0.0
        ]
        weights_list.sort(key=lambda x: x["weight"], reverse=True)
        return jsonify({"date": last_date, "weights": weights_list})
    except FileNotFoundError:
        return jsonify({"error": "no data yet"})


@app.route("/api/robustness")
def robustness():
    try:
        path = _latest(os.path.join(BASE, "data", "signals", "robust_results_*.csv"))
        df = pd.read_csv(path)
        df = df.sort_values("sharpe_ratio", ascending=False)
        # Round floats for cleaner JSON
        float_cols = ["cumulative_return", "annualised_return",
                      "annualised_volatility", "sharpe_ratio", "max_drawdown"]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
        return jsonify(df.to_dict(orient="records"))
    except FileNotFoundError:
        return jsonify({"error": "no data yet"})


@app.route("/api/logs")
def logs():
    try:
        path = os.path.join(BASE, "data", "live", "logs", "daily_execution.jsonl")
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        events = []
        for line in lines[-30:]:
            obj = json.loads(line)
            # Normalise: surface n_orders so the UI doesn't need to know the schema
            obj["n_orders"] = len(obj.get("orders", []))
            events.append(obj)
        events.reverse()  # most recent first
        return jsonify(events)
    except FileNotFoundError:
        return jsonify({"error": "no data yet"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
