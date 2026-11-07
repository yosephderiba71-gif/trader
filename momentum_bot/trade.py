"""CLI entrypoint: compute momentum targets, then rebalance Alpaca."""
from __future__ import annotations

import argparse

from .rebalance import rebalance
from .signals import (
    compute_metrics,
    compute_target_weights,
    fetch_prices,
    format_report,
    latest_prices,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-asset time-series momentum rebalance."
    )
    parser.add_argument("--period", default="2y", help="yfinance lookback period (default: 2y)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit orders to Alpaca; without this flag the run is a dry run.",
    )
    args = parser.parse_args(argv)

    prices = fetch_prices(period=args.period)
    metrics = compute_metrics(prices)
    targets = compute_target_weights(prices)

    print(format_report(metrics, targets))
    print("Mode: EXECUTE" if args.execute else "Mode: DRY RUN")
    rebalance(
        targets,
        latest_prices=latest_prices(prices),
        dry_run=not args.execute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
