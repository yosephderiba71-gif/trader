"""Command line entry point for the intraday sleeve.

Examples:
    python -m intraday_bot run --dry-run
    python -m intraday_bot status
    python -m intraday_bot backtest --csv intraday/spy_5m.csv
    python -m intraday_bot demo
    python -m intraday_bot dashboard --serve --open
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import run_backtest
from .config import IntradaySettings
from .dashboard import build_payload, serve, write_static
from .demo import make_bars, write_demo
from .journal import load, metrics
from .runner import read_state, run_cycle

ET = ZoneInfo("America/New_York")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_run(args) -> None:
    s = IntradaySettings.from_env()
    if args.now:
        now = datetime.fromisoformat(args.now)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
    if not args.loop:
        _print(run_cycle(s, now=now, dry_run=args.dry_run))
        return
    while True:
        result = run_cycle(s, dry_run=args.dry_run)
        state = result.get("state", {})
        print(
            f"{datetime.now(timezone.utc).isoformat()} "
            f"action={result.get('action', state.get('last_action', '?'))} "
            f"reason={result.get('reason', state.get('last_reason', '?'))} "
            f"price={state.get('price', '?')}"
        )
        time.sleep(max(args.interval, 15))


def cmd_status(args) -> None:
    s = IntradaySettings.from_env()
    state_path = "intraday/demo_state.json" if args.demo else (args.state or s.state_path)
    journal_path = "intraday/demo_journal.jsonl" if args.demo else (args.journal or s.journal_path)
    state = read_state(state_path)
    events = load(journal_path)
    _print({"state": state, "metrics": metrics(events, s.starting_equity)})


def cmd_backtest(args) -> None:
    s = IntradaySettings.from_env()
    if args.csv:
        frame = pd.read_csv(args.csv, index_col=0)
        raw_index = [str(value) for value in frame.index]
        tz_aware = any(re.search(r"(Z|[+-]\d{2}:?\d{2})$", value) for value in raw_index)
        if tz_aware:
            frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(ET)
        else:
            frame.index = pd.to_datetime(frame.index).tz_localize(ET)
        bars = frame[["open", "high", "low", "close", "volume"]].sort_index()
    else:
        bars = make_bars(sessions=args.sessions, seed=args.seed)
    result = run_backtest(bars, s, starting_equity=args.equity)
    summary = {
        "metrics": result.metrics,
        "trades": int(len(result.trades)),
        "sessions": result.sessions,
        "first": str(result.equity.index[0]) if not result.equity.empty else None,
        "last": str(result.equity.index[-1]) if not result.equity.empty else None,
    }
    _print(summary)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.trades.to_csv(target, index=False)
        print(f"Wrote {target}")


def cmd_fetch(args) -> None:
    from .broker import AlpacaBroker

    s = IntradaySettings.from_env()
    s.require_credentials()
    start = pd.Timestamp(args.start, tz=ET).to_pydatetime()
    end = pd.Timestamp(args.end, tz=ET).to_pydatetime()
    broker = AlpacaBroker(s.api_key, s.api_secret, paper=s.paper)
    bars = broker.bars(s.symbol, start, end, s.feed, s.bar_minutes, now=datetime.now(timezone.utc))
    if bars.empty:
        print("No bars returned.")
        return
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(target)
    print(f"Wrote {len(bars)} bars to {target}")


def cmd_demo(args) -> None:
    summary = write_demo(html_path=args.out)
    summary.pop("metrics", None)
    _print(summary)
    if args.open:
        import webbrowser

        webbrowser.open(Path(args.out).resolve().as_uri())


def cmd_dashboard(args) -> None:
    if args.serve:
        serve(
            "intraday/demo_journal.jsonl" if args.demo else args.journal,
            "intraday/demo_state.json" if args.demo else args.state,
            port=args.port,
            demo=args.demo,
            open_browser=args.open,
        )
        return
    payload = build_payload(
        "intraday/demo_journal.jsonl" if args.demo else args.journal,
        "intraday/demo_state.json" if args.demo else args.state,
        demo=args.demo,
    )
    target = write_static(payload, args.out)
    print(f"Wrote {target}")
    if args.open:
        import webbrowser

        webbrowser.open(target.resolve().as_uri())


def cmd_close(args) -> None:
    from .broker import AlpacaBroker
    from .runner import _close_and_log, _position_from
    from .bands import Bands

    s = IntradaySettings.from_env()
    s.require_credentials()
    state = read_state(s.state_path)
    broker = AlpacaBroker(s.api_key, s.api_secret, paper=s.paper)
    raw = broker.position(s.symbol)
    if raw is None:
        print("No open position.")
        return
    dummy = Bands(upper=0.0, lower=0.0, sigma=0.0, anchor_high=0.0, anchor_low=0.0)
    position = _position_from(raw, state, dummy, datetime.now(ET))
    if position is None:
        print("No open position.")
        return
    result = _close_and_log(
        broker,
        s,
        position,
        datetime.now(ET),
        datetime.now(ET).date(),
        "manual_close",
        float(getattr(raw, "current_price", 0.0) or 0.0),
        trade_id=(state.get("position") or {}).get("trade_id"),
    )
    if result.get("action") == "exit_long":
        state.pop("position", None)
    from .runner import write_state

    write_state(s.state_path, state)
    _print(result)


def main() -> None:
    parser = argparse.ArgumentParser(prog="intraday_bot", description="SPY intraday-momentum paper sleeve")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one paper cycle")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--loop", action="store_true", help="keep running every --interval seconds")
    run.add_argument("--interval", type=int, default=60)
    run.add_argument("--now", help="override now, ISO timestamp")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="print state and journal metrics")
    status.add_argument("--journal", help="journal JSONL path")
    status.add_argument("--state", help="state JSON path")
    status.add_argument("--demo", action="store_true", help="use the synthetic demo files")
    status.set_defaults(func=cmd_status)

    backtest = sub.add_parser("backtest", help="backtest 5-minute bars")
    backtest.add_argument("--csv", help="CSV with a datetime index and OHLCV columns")
    backtest.add_argument("--sessions", type=int, default=260, help="synthetic sessions when --csv is omitted")
    backtest.add_argument("--seed", type=int, default=11)
    backtest.add_argument("--equity", type=float, default=100_000.0)
    backtest.add_argument("--out", help="write trades CSV")
    backtest.set_defaults(func=cmd_backtest)

    fetch = sub.add_parser("fetch", help="download 5-minute bars to CSV")
    fetch.add_argument("--start", required=True, help="YYYY-MM-DD")
    fetch.add_argument("--end", required=True, help="YYYY-MM-DD")
    fetch.add_argument("--out", default="intraday/spy_5m.csv")
    fetch.set_defaults(func=cmd_fetch)

    demo = sub.add_parser("demo", help="build a synthetic demo dashboard")
    demo.add_argument("--out", default="intraday_demo.html")
    demo.add_argument("--open", action="store_true")
    demo.set_defaults(func=cmd_demo)

    dash = sub.add_parser("dashboard", help="build or serve the dashboard")
    dash.add_argument("--journal", default="intraday/journal.jsonl")
    dash.add_argument("--state", default="intraday/state.json")
    dash.add_argument("--demo", action="store_true")
    dash.add_argument("--serve", action="store_true")
    dash.add_argument("--port", type=int, default=8765)
    dash.add_argument("--out", default="intraday_dashboard.html")
    dash.add_argument("--open", action="store_true")
    dash.set_defaults(func=cmd_dashboard)

    close = sub.add_parser("close", help="flatten the paper position")
    close.set_defaults(func=cmd_close)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
