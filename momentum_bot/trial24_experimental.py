"""Experimental Trial #24 runner on its own isolated Alpaca paper account.

This mirrors ``momentum_bot.run_live`` but reads only the dedicated
``TRIAL24_ALPACA_API_KEY`` and ``TRIAL24_ALPACA_SECRET_KEY`` variables. It
never falls back to the main Discipline Bot credentials (``ALPACA_API_KEY`` /
``ALPACA_API_SECRET``), and it refuses to construct a trading client unless
``paper=True`` is asserted.

The script is a dry run unless ``--execute`` is passed.
"""
from __future__ import annotations

import argparse
import math
import os

import pandas as pd

from .rebalance import rebalance
from .sector_momentum import (
    SECTOR_ASSETS,
    SectorMetrics,
    compute_metrics,
    compute_target_weights,
)

BENCHMARK = "SPY"
ASSETS = SECTOR_ASSETS + (BENCHMARK,)

TRIAL24_API_KEY_ENV = "TRIAL24_ALPACA_API_KEY"
TRIAL24_API_SECRET_ENV = "TRIAL24_ALPACA_SECRET_KEY"


def assert_paper_mode(paper: bool) -> None:
    """Hard paper assertion: refuse anything that is not exactly True."""
    if paper is not True:
        raise ValueError(
            "Hard paper assertion failed: paper=True is required. "
            "Trial #24 is restricted to its dedicated paper account."
        )


def _trial24_credentials() -> tuple[str, str]:
    api_key = os.environ.get(TRIAL24_API_KEY_ENV)
    api_secret = os.environ.get(TRIAL24_API_SECRET_ENV)
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Trial #24 Alpaca credentials. Set "
            f"{TRIAL24_API_KEY_ENV} and {TRIAL24_API_SECRET_ENV}."
        )
    return api_key, api_secret


def build_trial24_clients(paper: bool = True):
    """Build trading and data clients with only the dedicated Trial #24 keys."""
    assert_paper_mode(paper)
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    api_key, api_secret = _trial24_credentials()
    trading = TradingClient(api_key, api_secret, paper=paper)
    data = StockHistoricalDataClient(api_key, api_secret)
    return trading, data


def fetch_prices(period: str = "2y") -> pd.DataFrame:
    """Download adjusted daily closes for the 11 sectors plus SPY."""
    import yfinance as yf

    raw = yf.download(
        list(ASSETS),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no price data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        if len(ASSETS) == 1:
            close.columns = [ASSETS[0]]

    missing = [symbol for symbol in ASSETS if symbol not in close.columns]
    if missing:
        raise RuntimeError(f"Missing price data for: {', '.join(missing)}")

    return close.loc[:, list(ASSETS)].sort_index()


def latest_prices(prices: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for symbol in ASSETS:
        if symbol not in prices.columns:
            continue
        series = pd.to_numeric(prices[symbol], errors="coerce").dropna()
        if not series.empty:
            result[symbol] = float(series.iloc[-1])
    return result


def _format_number(value: float, spec: str) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:{spec}}"


def format_report(
    metrics: list[SectorMetrics],
    targets: dict[str, float],
    benchmark_close: float | None = None,
) -> str:
    lines = ["Trial #24 experimental sector momentum signal snapshot:"]
    lines.append(
        f"{'SYMBOL':<8}{'CLOSE':>10}{'SMA200':>10}{'ELIGIBLE':>10}{'120D SKIP':>12}"
    )
    for metric in metrics:
        lines.append(
            f"{metric.symbol:<8}{_format_number(metric.close, '.2f'):>10}"
            f"{_format_number(metric.sma200, '.2f'):>10}"
            f"{str(metric.eligible):>10}"
            f"{_format_number(metric.skip_return_120d, '.2%'):>12}"
        )
    incomplete = [
        metric.symbol
        for metric in metrics
        if not (
            math.isfinite(metric.close)
            and math.isfinite(metric.sma200)
            and math.isfinite(metric.skip_return_120d)
        )
    ]
    if incomplete:
        lines.append(
            "WARNING: missing price data for "
            + ", ".join(incomplete)
            + "; treated as ineligible."
        )
    lines.append(
        "Target weights: "
        + ", ".join(f"{symbol} {weight:.2%}" for symbol, weight in targets.items())
    )
    if benchmark_close is not None:
        lines.append(f"{BENCHMARK} benchmark close: {benchmark_close:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trial #24 experimental runner on an isolated Alpaca paper account."
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="yfinance lookback period (default: 2y)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit orders to the dedicated Trial #24 paper account.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request a dry run (the default when --execute is absent).",
    )
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")

    trading_client, data_client = build_trial24_clients(paper=True)

    prices = fetch_prices(period=args.period)
    metrics = compute_metrics(prices)
    targets = compute_target_weights(prices)
    prices_by_symbol = latest_prices(prices)
    benchmark_close = prices_by_symbol.get(BENCHMARK)

    print(format_report(metrics, targets, benchmark_close))
    print("Mode: EXECUTE" if args.execute else "Mode: DRY RUN")
    rebalance(
        targets,
        latest_prices=prices_by_symbol,
        dry_run=not args.execute,
        trading_client=trading_client,
        data_client=data_client,
        require_first_trading_day=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
