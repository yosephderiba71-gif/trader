"""Command line.

  python -m trader run              dry run: decides and logs, places no orders
  python -m trader run --execute    places paper orders
  python -m trader research         out-of-sample results per symbol and strategy
  python -m trader status           account, positions
"""
from __future__ import annotations

import argparse

from .config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trader", description="Alpaca paper-trading agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run one trading cycle (dry run unless --execute)")
    run_parser.add_argument("--execute", action="store_true", help="submit paper orders")
    sub.add_parser("research", help="show out-of-sample results per symbol and strategy")
    sub.add_parser("status", help="show account and positions")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    settings.require_credentials()

    from .agent import log, run_cycle
    from .broker import AlpacaBroker
    from .data import fetch_bars
    from .research import evaluate_all, qualifying_best, report_frame

    broker = AlpacaBroker(settings.api_key, settings.api_secret)

    if args.command == "status":
        account = broker.account()
        print(f"equity={account.equity} buying_power={account.buying_power} last_equity={account.last_equity}")
        for symbol, qty in broker.positions().items():
            print(f"  {symbol}: {qty}")
        return 0

    bars = fetch_bars(broker.data, settings.universe, settings)

    if args.command == "research":
        candidates = evaluate_all(bars, settings)
        print(report_frame(candidates).to_string(index=False))
        print("\nQualifying strategy per symbol:")
        for candidate in qualifying_best(candidates, settings).values():
            print(f"  {candidate.symbol}: {candidate.strategy}")
        return 0

    report = run_cycle(settings, broker, bars, execute=args.execute)
    log("cycle_done", executed=args.execute, actions=len(report.actions), halted=report.halted, skipped=report.skipped)
    return 0
