"""Build a QuantStats HTML report from Alpaca paper-account portfolio history.

Read-only: fetches portfolio history and writes reports/report.html.
Credentials are read only from ALPACA_API_KEY / ALPACA_API_SECRET.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import quantstats as qs
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

REPORT_PATH = Path(__file__).resolve().parent / "reports" / "report.html"


def daily_returns(period: str = "1A", timeframe: str = "1D") -> pd.Series:
    """Fetch paper-account equity history and return daily returns."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    if not (api_key and api_secret):
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_API_SECRET (use paper-trading keys).")

    client = TradingClient(api_key, api_secret, paper=True)
    history = client.get_portfolio_history(
        history_filter=GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
    )

    index = pd.to_datetime(history.timestamp, unit="s", utc=True)
    equity = pd.Series(history.equity, index=index, dtype="float64").dropna()
    equity = equity[equity > 0]
    equity = equity[~equity.index.duplicated(keep="last")]
    if len(equity) < 2:
        raise SystemExit("Not enough portfolio history to compute daily returns.")

    returns = equity.pct_change().dropna()
    returns.index = returns.index.tz_localize(None)
    return returns


def main() -> None:
    returns = daily_returns()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        returns,
        output=str(REPORT_PATH),
        title="Trading Agent - Paper Account",
    )
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
