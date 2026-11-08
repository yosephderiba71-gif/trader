"""Live entrypoint for the Trial #24 unlevered sector momentum strategy.

The script fetches daily closes for the 11 SPDR sectors plus SPY, computes the
frozen Trial #24 signals, prints the target weights, and hands execution to
``momentum_bot.rebalance``. It is a dry run unless ``--execute`` is passed.
"""
from __future__ import annotations

import argparse
import math

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
    lines = ["Trial #24 sector momentum signal snapshot:"]
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
        description="Trial #24 unlevered sector momentum live rebalance."
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="yfinance lookback period (default: 2y)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit orders to Alpaca; without this flag the run is a dry run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request a dry run (the default when --execute is absent).",
    )
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")

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
        require_first_trading_day=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
