"""Legacy SPY-only dashboard for the intraday sleeve.

The canonical read-only multi-asset viewer is ``market_viewer.py``. This module
remains because ``intraday_bot.cli`` and the intraday demo still import it.

No Streamlit, no CDN, no build step. It reads the append-only journal and
``state.json``, embeds the data into one HTML file, and can optionally serve it
with a tiny standard-library HTTP server that refreshes every 30 seconds.

    python -m intraday_bot.dashboard --demo --open
    python -m intraday_bot.dashboard --serve --port 8765
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .journal import (
    closed_trades,
    daily_pnl,
    equity_curve,
    load,
    metrics,
    open_trade as open_entry,
    trade_rows,
)


def _read_json(path: str | Path) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        obj = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def build_payload(
    journal_path: str | Path = "intraday/journal.jsonl",
    state_path: str | Path = "intraday/state.json",
    starting_equity: float | None = None,
    demo: bool = False,
    live: bool = False,
) -> dict:
    events = load(journal_path)
    state = _read_json(state_path)
    start_eq = _float(state.get("starting_equity"), starting_equity or 100_000.0)
    m = metrics(events, start_eq)
    rows = trade_rows(events)
    account_equity = _float(state.get("equity"), start_eq + m["net_pnl"])
    today = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()

    position = state.get("position")
    open_trade = None
    if isinstance(position, dict) and _float(position.get("qty")) > 0:
        qty = int(_float(position.get("qty")))
        entry = _float(position.get("entry_price"))
        current = _float(position.get("current_price"), _float(state.get("price"), entry))
        stop = _float(position.get("stop"))
        unrealized = _float(position.get("unrealized_pl"), (current - entry) * qty)
        open_trade = {
            "symbol": position.get("symbol") or state.get("symbol") or "SPY",
            "side": position.get("side") or "long",
            "qty": qty,
            "entry_price": entry,
            "current_price": current,
            "stop": stop,
            "band": _float(position.get("band")),
            "vwap": _float(position.get("vwap")),
            "unrealized_pl": unrealized,
            "unrealized_pct": (current / entry - 1.0) if entry > 0 else 0.0,
            "distance_to_stop_pct": (current - stop) / current if current > 0 and stop > 0 else 0.0,
            "entry_ts": position.get("entry_ts"),
            "trade_id": position.get("trade_id"),
        }
    else:
        entry_event = open_entry(events)
        if entry_event:
            qty = int(_float(entry_event.get("qty")))
            entry = _float(entry_event.get("price"))
            current = _float(state.get("price"), entry)
            stop = _float(entry_event.get("stop"))
            open_trade = {
                "symbol": entry_event.get("symbol", "SPY"),
                "side": "long",
                "qty": qty,
                "entry_price": entry,
                "current_price": current,
                "stop": stop,
                "band": _float(entry_event.get("upper")),
                "vwap": _float(entry_event.get("vwap")),
                "unrealized_pl": (current - entry) * qty,
                "unrealized_pct": (current / entry - 1.0) if entry > 0 else 0.0,
                "distance_to_stop_pct": (current - stop) / current if current > 0 and stop > 0 else 0.0,
                "entry_ts": entry_event.get("entry_ts") or entry_event.get("ts"),
                "trade_id": entry_event.get("trade_id"),
            }

    realized_today = _float(
        state.get("realized_today"),
        sum(t["pnl"] for t in rows if t["date"] == today),
    )
    today_pnl = _float(state.get("today_pnl"), realized_today + (open_trade["unrealized_pl"] if open_trade else 0.0))
    open_pnl = open_trade["unrealized_pl"] if open_trade else 0.0
    losses = [t for t in rows if t["pnl"] < 0]
    worst = sorted(losses, key=lambda t: t["pnl"])[:6]
    by_reason: dict[str, float] = {}
    for t in losses:
        by_reason[t["reason"]] = by_reason.get(t["reason"], 0.0) + t["pnl"]

    market_open = state.get("market_open")
    if demo:
        status, status_class = "Demo", "amber"
    elif state.get("error"):
        status, status_class = "Error", "bad"
    elif market_open is True:
        status, status_class = "Market open", "ok"
    elif market_open is False:
        status, status_class = "Market closed", "idle"
    else:
        status, status_class = "No run yet", "idle"

    health = {
        "status": status,
        "status_class": status_class,
        "market_open": market_open,
        "last_run": state.get("updated_at"),
        "last_bar": state.get("last_bar"),
        "data_age_min": state.get("data_age_min"),
        "stale_data": bool(state.get("stale_data")),
        "daily_loss_halt": bool(state.get("daily_loss_halt")),
        "price": state.get("price"),
        "open": state.get("open"),
        "prior_close": state.get("prior_close"),
        "upper": state.get("upper"),
        "lower": state.get("lower"),
        "vwap": state.get("vwap"),
        "sigma": state.get("sigma"),
        "trades_today": state.get("trades_today"),
        "last_action": state.get("last_action"),
        "last_reason": state.get("last_reason"),
        "sessions_available": state.get("sessions_available"),
        "sessions_required": state.get("sessions_required"),
        "error": state.get("error"),
    }
    strategy = state.get("strategy") or {
        "name": "SPY intraday momentum (long-only)",
        "band_lookback_sessions": 14,
        "band_mult": 1.0,
        "risk_per_trade": 0.0025,
        "max_position_pct": 1.0,
        "max_trades_per_day": 1,
        "max_daily_loss_pct": 0.01,
        "flat_time": "15:45",
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {"demo": demo, "live": live, "version": "0.1.0"},
        "bot": {
            "name": "SPY Intraday Momentum",
            "symbol": state.get("symbol") or "SPY",
            "mode": "long-only",
            "paper": True,
        },
        "account": {
            "equity": account_equity,
            "starting_equity": start_eq,
            "today_pnl": today_pnl,
            "realized_today": realized_today,
            "open_pnl": open_pnl,
        },
        "metrics": {
            **m,
            "today_pnl": today_pnl,
            "realized_today": realized_today,
            "open_pnl": open_pnl,
        },
        "open_trade": open_trade,
        "equity_curve": equity_curve(events, start_eq),
        "daily_pnl": daily_pnl(events),
        "trades": rows,
        "losses": {
            "count": len(losses),
            "total": sum(t["pnl"] for t in losses),
            "avg": (sum(t["pnl"] for t in losses) / len(losses)) if losses else 0.0,
            "largest": min((t["pnl"] for t in losses), default=0.0),
            "streak": m["loss_streak"],
            "by_reason": by_reason,
            "worst": worst,
        },
        "health": health,
        "strategy": strategy,
        "notes": [
            "Paper trading only. No live orders are sent by this dashboard.",
            "Long-only adaptation; the short side of the paper is intentionally disabled.",
            "The mandatory flat exit is 15:45 ET so it fits a 15-minute scheduler.",
            "A stop is checked at each half-hour decision point; entries only happen there too.",
            "Backtest fills are next-bar opens with slippage on both sides.",
        ],
    }


def render_html(payload: dict) -> str:
    data = json.dumps(payload, default=str).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data)


def write_static(payload: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(payload), encoding="utf-8")
    return target


def serve(
    journal_path: str | Path,
    state_path: str | Path,
    port: int = 8765,
    demo: bool = False,
    open_browser: bool = False,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                payload = build_payload(journal_path, state_path, demo=demo, live=True)
                body = render_html(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/api/data"):
                payload = build_payload(journal_path, state_path, demo=demo, live=True)
                body = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    url = f"http://127.0.0.1:{port}"
    print(f"Dashboard: {url}")
    if open_browser:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday sleeve dashboard")
    parser.add_argument("--journal", default="intraday/journal.jsonl")
    parser.add_argument("--state", default="intraday/state.json")
    parser.add_argument("--demo", action="store_true", help="use intraday/demo_* files")
    parser.add_argument("--serve", action="store_true", help="serve a live local page")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    parser.add_argument("--out", default="intraday_dashboard.html", help="static HTML output path")
    args = parser.parse_args()

    journal = "intraday/demo_journal.jsonl" if args.demo else args.journal
    state = "intraday/demo_state.json" if args.demo else args.state

    if args.serve:
        serve(journal, state, port=args.port, demo=args.demo, open_browser=args.open)
        return

    payload = build_payload(journal, state, demo=args.demo)
    target = write_static(payload, args.out)
    print(f"Wrote {target}")
    if args.open:
        webbrowser.open(target.resolve().as_uri())


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPY Intraday Momentum</title>
<style>
:root {
  --bg: #0a0e16;
  --panel: #111827;
  --panel-2: #0d1422;
  --line: #1f2a3a;
  --line-soft: #172233;
  --text: #e6edf7;
  --muted: #8b9bb4;
  --muted-2: #61708a;
  --green: #34d399;
  --green-soft: rgba(52, 211, 153, .12);
  --red: #f87171;
  --red-soft: rgba(248, 113, 113, .12);
  --amber: #fbbf24;
  --blue: #60a5fa;
  --violet: #a78bfa;
  --radius: 16px;
  --shadow: 0 12px 40px rgba(0, 0, 0, .28);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background:
    radial-gradient(1100px 500px at 15% -10%, rgba(96, 165, 250, .10), transparent 60%),
    radial-gradient(900px 420px at 100% 0%, rgba(167, 139, 250, .08), transparent 55%),
    var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  min-height: 100vh;
}
a { color: var(--blue); text-decoration: none; }
.wrap { max-width: 1240px; margin: 0 auto; padding: 24px 22px 48px; }
.topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 8px 0 22px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  width: 42px; height: 42px; border-radius: 12px;
  display: grid; place-items: center;
  font-weight: 800; font-size: 12px; letter-spacing: .08em;
  color: #06121f;
  background: linear-gradient(135deg, #67e8f9, #818cf8);
  box-shadow: 0 8px 24px rgba(129, 140, 248, .25);
}
.brand h1 { font-size: 18px; line-height: 1.15; margin: 0; letter-spacing: -.01em; }
.brand p { margin: 3px 0 0; color: var(--muted); font-size: 12px; }
.top-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.muted { color: var(--muted); }
.tiny { font-size: 12px; }
.pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 11px; border-radius: 999px;
  border: 1px solid var(--line);
  background: var(--panel-2);
  font-size: 12px; font-weight: 600; color: var(--muted);
}
.pill::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--muted-2); }
.pill.ok { color: var(--green); border-color: rgba(52, 211, 153, .28); background: var(--green-soft); }
.pill.ok::before { background: var(--green); box-shadow: 0 0 0 4px rgba(52, 211, 153, .10); }
.pill.bad { color: var(--red); border-color: rgba(248, 113, 113, .28); background: var(--red-soft); }
.pill.bad::before { background: var(--red); }
.pill.idle { color: var(--amber); border-color: rgba(251, 191, 36, .25); background: rgba(251, 191, 36, .08); }
.pill.idle::before { background: var(--amber); }
.btn {
  appearance: none; border: 1px solid var(--line); background: var(--panel);
  color: var(--text); border-radius: 10px; padding: 8px 12px;
  font: inherit; font-size: 12px; font-weight: 600; cursor: pointer;
  transition: border-color .15s ease, background .15s ease;
}
.btn:hover { border-color: #334155; background: #151f30; }
.btn:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.demo-banner {
  display: flex; align-items: center; gap: 10px;
  border: 1px solid rgba(96, 165, 250, .25);
  background: rgba(96, 165, 250, .08);
  color: #bfdbfe; border-radius: 12px; padding: 10px 14px;
  font-size: 12px; margin-bottom: 18px;
}
.demo-banner strong { color: #dbeafe; }
.hero {
  display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px;
  align-items: end; padding: 26px 28px; margin-bottom: 16px;
  border: 1px solid var(--line); border-radius: 20px;
  background: linear-gradient(135deg, rgba(17, 24, 39, .96), rgba(13, 20, 34, .96));
  box-shadow: var(--shadow);
}
.eyebrow { color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .14em; }
.hero-value { font-size: 46px; font-weight: 800; letter-spacing: -.035em; line-height: 1.05; margin: 8px 0 4px; }
.hero-sub { color: var(--muted); font-size: 13px; }
.hero-side { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.mini {
  border: 1px solid var(--line-soft); background: var(--panel-2);
  border-radius: 12px; padding: 12px;
}
.mini span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 5px; }
.mini strong { font-size: 15px; font-weight: 700; }
.kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 16px; }
.kpi {
  border: 1px solid var(--line); background: var(--panel);
  border-radius: 14px; padding: 14px 15px;
}
.kpi span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 7px; }
.kpi strong { display: block; font-size: 19px; letter-spacing: -.02em; }
.kpi small { display: block; color: var(--muted-2); font-size: 11px; margin-top: 4px; }
.grid-2 { display: grid; grid-template-columns: 1.35fr 1fr; gap: 16px; margin-bottom: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.card {
  border: 1px solid var(--line); background: var(--panel);
  border-radius: var(--radius); padding: 18px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .14);
}
.card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
.card h2 { font-size: 13px; margin: 0; letter-spacing: .01em; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; font-weight: 700; padding: 4px 8px; border-radius: 999px;
  background: var(--panel-2); border: 1px solid var(--line); color: var(--muted);
}
.badge.green { color: var(--green); background: var(--green-soft); border-color: rgba(52, 211, 153, .25); }
.badge.red { color: var(--red); background: var(--red-soft); border-color: rgba(248, 113, 113, .25); }
.badge.amber { color: var(--amber); background: rgba(251, 191, 36, .08); border-color: rgba(251, 191, 36, .22); }
.chart { min-height: 250px; }
.chart svg { width: 100%; height: auto; display: block; }
.chart-empty {
  min-height: 220px; display: grid; place-items: center; text-align: center;
  color: var(--muted); border: 1px dashed var(--line); border-radius: 12px;
  padding: 22px; font-size: 13px;
}
.open-trade { display: grid; gap: 12px; }
.open-top { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
.open-symbol { font-size: 24px; font-weight: 800; letter-spacing: -.03em; }
.open-side { color: var(--green); font-size: 11px; font-weight: 700; letter-spacing: .12em; }
.open-pnl { font-size: 22px; font-weight: 800; text-align: right; }
.open-pnl small { display: block; font-size: 11px; font-weight: 500; color: var(--muted); }
.open-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
.open-grid div { background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 10px; padding: 9px 10px; }
.open-grid span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.open-grid strong { font-size: 13px; }
.cushion { margin-top: 2px; }
.cushion-labels { display: flex; justify-content: space-between; color: var(--muted); font-size: 10px; margin-bottom: 5px; }
.cushion-track { height: 7px; border-radius: 999px; background: #1b2638; overflow: hidden; }
.cushion-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--red), var(--amber), var(--green)); }
.loss-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-bottom: 12px; }
.loss-summary div { background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 10px; padding: 9px 10px; }
.loss-summary span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.loss-summary strong { font-size: 15px; }
.worst { display: grid; gap: 7px; }
.worst-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 7px 0; border-top: 1px solid var(--line-soft); font-size: 12px; }
.worst-row:first-child { border-top: 0; }
.worst-row span { color: var(--muted); }
.health { display: grid; gap: 8px; }
.health-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 12px; padding: 6px 0; border-bottom: 1px solid var(--line-soft); }
.health-row:last-child { border-bottom: 0; }
.health-row span { color: var(--muted); }
.health-row strong { font-weight: 600; text-align: right; }
.alert {
  border: 1px solid rgba(248, 113, 113, .28); background: var(--red-soft);
  color: #fecaca; border-radius: 10px; padding: 9px 11px; font-size: 12px; margin-top: 8px;
}
.alert.warn { border-color: rgba(251, 191, 36, .25); background: rgba(251, 191, 36, .08); color: #fde68a; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 16px; }
.tab {
  appearance: none; border: 1px solid transparent; background: transparent;
  color: var(--muted); border-radius: 9px; padding: 7px 11px;
  font: inherit; font-size: 12px; font-weight: 600; cursor: pointer;
}
.tab:hover { color: var(--text); background: var(--panel-2); }
.tab.active { color: var(--text); background: var(--panel-2); border-color: var(--line); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.overview-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px; }
.how { display: grid; gap: 10px; }
.step { display: flex; gap: 11px; align-items: flex-start; }
.step-num {
  flex: 0 0 auto; width: 24px; height: 24px; border-radius: 8px;
  display: grid; place-items: center; font-size: 11px; font-weight: 800;
  color: #06121f; background: linear-gradient(135deg, #67e8f9, #818cf8);
}
.step p { margin: 2px 0 0; color: var(--muted); font-size: 12px; }
.band-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
.band-grid div { background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 10px; padding: 10px; }
.band-grid span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.band-grid strong { font-size: 14px; }
.notes { margin: 16px 0 0; padding: 0; list-style: none; display: grid; gap: 7px; }
.notes li { color: var(--muted); font-size: 12px; padding-left: 14px; position: relative; }
.notes li::before { content: ""; position: absolute; left: 0; top: 7px; width: 5px; height: 5px; border-radius: 50%; background: var(--muted-2); }
.toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
.chip {
  appearance: none; border: 1px solid var(--line); background: var(--panel-2);
  color: var(--muted); border-radius: 999px; padding: 6px 10px;
  font: inherit; font-size: 11px; font-weight: 600; cursor: pointer;
}
.chip.active { color: var(--text); border-color: #334155; background: #172033; }
.search {
  flex: 1 1 180px; min-width: 150px; border: 1px solid var(--line);
  background: var(--panel-2); color: var(--text); border-radius: 9px;
  padding: 7px 10px; font: inherit; font-size: 12px;
}
.search::placeholder { color: var(--muted-2); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; padding: 9px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 10px; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
tr:hover td { background: rgba(255, 255, 255, .015); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.green { color: var(--green); }
.red { color: var(--red); }
.amber { color: var(--amber); }
.dim { color: var(--muted); }
.reason { color: var(--muted); font-size: 11px; }
.empty {
  min-height: 160px; display: grid; place-items: center; text-align: center;
  color: var(--muted); border: 1px dashed var(--line); border-radius: 12px; padding: 22px;
}
.empty strong { display: block; color: var(--text); margin-bottom: 4px; }
.settings-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.setting { background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 10px; padding: 11px 12px; }
.setting span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.setting strong { font-size: 14px; }
.footer { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; color: var(--muted-2); font-size: 11px; padding-top: 22px; }
.footer a { color: var(--muted); }
@media (max-width: 980px) {
  .kpis { grid-template-columns: repeat(3, 1fr); }
  .grid-2, .grid-3, .overview-grid { grid-template-columns: 1fr; }
  .hero { grid-template-columns: 1fr; }
  .settings-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 620px) {
  .wrap { padding: 16px 14px 36px; }
  .topbar { align-items: flex-start; flex-direction: column; }
  .hero { padding: 20px; }
  .hero-value { font-size: 36px; }
  .hero-side { grid-template-columns: 1fr 1fr; }
  .kpis { grid-template-columns: 1fr 1fr; }
  .settings-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="wrap">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">SPY</div>
      <div>
        <h1>Intraday Momentum</h1>
        <p>Paper trading sleeve · long-only · flat by close</p>
      </div>
    </div>
    <div class="top-right">
      <span id="statusPill" class="pill idle">Loading</span>
      <span id="updated" class="muted tiny">—</span>
      <button class="btn" onclick="location.reload()">Refresh</button>
    </div>
  </header>

  <div id="demoBanner" class="demo-banner" style="display:none">
    <strong>Demo data.</strong> This page is showing a synthetic sample so you can see the layout. Run the paper bot to replace it with real journal data.
  </div>

  <section class="hero">
    <div>
      <div class="eyebrow">NET P&amp;L</div>
      <div id="netPnl" class="hero-value">$0.00</div>
      <div id="netSub" class="hero-sub">No closed trades yet</div>
    </div>
    <div class="hero-side">
      <div class="mini"><span>Today</span><strong id="todayPnl">$0.00</strong></div>
      <div class="mini"><span>Open P&amp;L</span><strong id="openPnl">$0.00</strong></div>
      <div class="mini"><span>Win rate</span><strong id="heroWinRate">—</strong></div>
    </div>
  </section>

  <section class="kpis">
    <div class="kpi"><span>Win rate</span><strong id="kpiWinRate">—</strong><small id="kpiWinSub">0 wins / 0 losses</small></div>
    <div class="kpi"><span>Profit factor</span><strong id="kpiPf">—</strong><small>gross win / gross loss</small></div>
    <div class="kpi"><span>Max drawdown</span><strong id="kpiDd">—</strong><small>closed-trade equity</small></div>
    <div class="kpi"><span>Trades</span><strong id="kpiTrades">0</strong><small id="kpiTradeSub">0 trade days</small></div>
    <div class="kpi"><span>Avg R</span><strong id="kpiR">—</strong><small id="kpiRSub">expectancy $0.00</small></div>
    <div class="kpi"><span>Sharpe</span><strong id="kpiSharpe">—</strong><small>daily, active days</small></div>
  </section>

  <section class="grid-2">
    <div class="card">
      <div class="card-head"><h2>Equity curve</h2><span id="equityMeta" class="muted tiny">—</span></div>
      <div id="equityChart" class="chart"></div>
    </div>
    <div class="card">
      <div class="card-head"><h2>Daily P&amp;L</h2><span id="dailyMeta" class="muted tiny">—</span></div>
      <div id="dailyChart" class="chart"></div>
    </div>
  </section>

  <section class="grid-3">
    <div class="card">
      <div class="card-head"><h2>Open trade</h2><span id="openBadge" class="badge">Flat</span></div>
      <div id="openTrade"></div>
    </div>
    <div class="card">
      <div class="card-head"><h2>Losses</h2><span class="muted tiny">closed trades</span></div>
      <div id="lossSummary"></div>
      <div id="worstLosses"></div>
    </div>
    <div class="card">
      <div class="card-head"><h2>Bot health</h2><span id="healthBadge" class="badge">—</span></div>
      <div id="health"></div>
    </div>
  </section>

  <section class="card">
    <div class="tabs">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="trades">Trades</button>
      <button class="tab" data-tab="losses">Loss analysis</button>
      <button class="tab" data-tab="risk">Risk &amp; settings</button>
    </div>

    <div id="tab-overview" class="tab-panel active">
      <div class="overview-grid">
        <div>
          <h2 style="margin-bottom:12px">How the sleeve works</h2>
          <div class="how">
            <div class="step"><div class="step-num">1</div><div><strong>Build the noise area</strong><p>For each half-hour slot, average SPY's absolute move from the open over the prior 14 sessions. Bands are anchored on today's open and yesterday's close.</p></div></div>
            <div class="step"><div class="step-num">2</div><div><strong>Trade the break</strong><p>At 10:00, 10:30, … 15:30 ET, go long only when price is above the upper band. One trade per day.</p></div></div>
            <div class="step"><div class="step-num">3</div><div><strong>Trail and flatten</strong><p>The stop is the higher of the upper band and VWAP, checked at each half-hour decision point. Everything is flat by 15:45 ET.</p></div></div>
          </div>
          <ul id="notes" class="notes"></ul>
        </div>
        <div>
          <h2 style="margin-bottom:12px">Today's bands</h2>
          <div id="bandGrid" class="band-grid"></div>
          <div id="bandNote" class="muted tiny" style="margin-top:10px"></div>
        </div>
      </div>
    </div>

    <div id="tab-trades" class="tab-panel">
      <div class="toolbar">
        <button class="chip active" data-filter="all">All</button>
        <button class="chip" data-filter="wins">Wins</button>
        <button class="chip" data-filter="losses">Losses</button>
        <input id="tradeSearch" class="search" placeholder="Filter by date, reason, or trade id">
      </div>
      <div id="tradeTable" class="table-wrap"></div>
    </div>

    <div id="tab-losses" class="tab-panel">
      <div id="lossKpis" class="kpis" style="grid-template-columns:repeat(4,1fr)"></div>
      <div class="grid-2" style="margin-top:14px">
        <div>
          <h2 style="margin-bottom:10px">Worst closed trades</h2>
          <div id="lossTable" class="table-wrap"></div>
        </div>
        <div>
          <h2 style="margin-bottom:10px">Losses by exit reason</h2>
          <div id="lossReasons"></div>
        </div>
      </div>
    </div>

    <div id="tab-risk" class="tab-panel">
      <div id="settings" class="settings-grid"></div>
      <div id="riskAlerts" style="margin-top:12px"></div>
    </div>
  </section>

  <footer class="footer">
    <span>Paper trading only. Not financial advice. No live orders are sent by this page.</span>
    <span id="footerMeta">—</span>
  </footer>
</div>

<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const money0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

function fmtMoney(v, signed = true) {
  const n = Number(v || 0);
  const s = money.format(Math.abs(n));
  if (!signed) return (n < 0 ? "-" : "") + s;
  return (n > 0 ? "+" : n < 0 ? "-" : "") + s;
}
function fmtPct(v, digits = 2, signed = true) {
  const n = Number(v || 0) * 100;
  const s = Math.abs(n).toFixed(digits) + "%";
  if (!signed) return (n < 0 ? "-" : "") + s;
  return (n > 0 ? "+" : n < 0 ? "-" : "") + s;
}
function fmtNum(v, digits = 2) {
  const n = Number(v || 0);
  return n.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
function fmtR(v) {
  const n = Number(v || 0);
  return (n > 0 ? "+" : "") + n.toFixed(2) + "R";
}
function tone(v) { return Number(v) > 0 ? "green" : Number(v) < 0 ? "red" : "dim"; }
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function dateOnly(s) { return s ? String(s).slice(0, 10) : "—"; }
const ET_TZ = "America/New_York";
function etTime(s, withSeconds = false) {
  if (!s) return "—";
  const d = new Date(s);
  if (isNaN(d.getTime())) return String(s).slice(11, 16);
  const opts = { timeZone: ET_TZ, hour: "2-digit", minute: "2-digit", hour12: false };
  if (withSeconds) opts.second = "2-digit";
  return d.toLocaleTimeString("en-US", opts) + " ET";
}
function etDateTime(s) {
  if (!s) return "—";
  const d = new Date(s);
  if (isNaN(d.getTime())) return String(s);
  return d.toLocaleString("en-US", { timeZone: ET_TZ, month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) + " ET";
}
function timeOnly(s) { return etTime(s); }
function empty(title, body) {
  return `<div class="empty"><div><strong>${esc(title)}</strong>${esc(body || "")}</div></div>`;
}

function renderHeader() {
  const h = DATA.health;
  const pill = $("statusPill");
  pill.textContent = h.status;
  pill.className = "pill " + (h.status_class || "idle");
  $("updated").textContent = DATA.generated_at ? "Updated " + etTime(DATA.generated_at, true) : "—";
  $("demoBanner").style.display = DATA.meta.demo ? "flex" : "none";
  $("footerMeta").textContent = `${DATA.bot.name} · ${DATA.bot.symbol} · paper`;
}

function renderHero() {
  const m = DATA.metrics;
  const a = DATA.account;
  $("netPnl").textContent = fmtMoney(m.net_pnl);
  $("netPnl").className = "hero-value " + tone(m.net_pnl);
  $("netSub").textContent = `${fmtPct(m.total_return_pct)} on ${money0.format(a.starting_equity)} · ${m.trades} closed trade${m.trades === 1 ? "" : "s"}`;
  $("todayPnl").textContent = fmtMoney(a.today_pnl);
  $("todayPnl").className = tone(a.today_pnl);
  $("openPnl").textContent = fmtMoney(a.open_pnl);
  $("openPnl").className = tone(a.open_pnl);
  $("heroWinRate").textContent = m.trades ? fmtPct(m.win_rate, 1, false) : "—";

  $("kpiWinRate").textContent = m.trades ? fmtPct(m.win_rate, 1, false) : "—";
  $("kpiWinSub").textContent = `${m.wins} wins / ${m.losses} losses`;
  $("kpiPf").textContent = m.profit_factor == null ? "—" : fmtNum(m.profit_factor, 2);
  $("kpiDd").textContent = fmtPct(m.max_drawdown_pct);
  $("kpiDd").className = m.max_drawdown_pct < 0 ? "red" : "";
  $("kpiTrades").textContent = String(m.trades);
  $("kpiTradeSub").textContent = `${m.sessions || 0} trade days · ${fmtNum(m.avg_hold_minutes, 0)} min avg hold`;
  $("kpiR").textContent = m.trades ? fmtR(m.avg_r) : "—";
  $("kpiR").className = tone(m.avg_r);
  $("kpiRSub").textContent = `expectancy ${fmtMoney(m.expectancy)}`;
  $("kpiSharpe").textContent = m.trades ? fmtNum(m.sharpe, 2) : "—";
  $("kpiSharpe").className = tone(m.sharpe);
}

function renderOpen() {
  const t = DATA.open_trade;
  const badge = $("openBadge");
  const box = $("openTrade");
  if (!t) {
    badge.textContent = "Flat";
    badge.className = "badge";
    box.innerHTML = empty("No open trade", "Waiting for a close above the upper band at a half-hourly check.");
    return;
  }
  badge.textContent = "Long";
  badge.className = "badge green";
  const pnlTone = tone(t.unrealized_pl);
  const range = t.entry_price - t.stop;
  const progress = range > 0 ? Math.max(0, Math.min(100, ((t.current_price - t.stop) / range) * 100)) : 0;
  box.innerHTML = `
    <div class="open-trade">
      <div class="open-top">
        <div><div class="open-symbol">${esc(t.symbol)}</div><div class="open-side">LONG · ${t.qty} SHARES</div></div>
        <div class="open-pnl ${pnlTone}">${fmtMoney(t.unrealized_pl)}<small>${fmtPct(t.unrealized_pct)} unrealized</small></div>
      </div>
      <div class="open-grid">
        <div><span>Entry</span><strong>${money.format(t.entry_price)}</strong></div>
        <div><span>Current</span><strong>${money.format(t.current_price)}</strong></div>
        <div><span>Stop</span><strong class="red">${money.format(t.stop)}</strong></div>
        <div><span>Stop cushion</span><strong>${fmtPct(t.distance_to_stop_pct, 2, false)}</strong></div>
      </div>
      <div class="cushion">
        <div class="cushion-labels"><span>stop ${money.format(t.stop)}</span><span>entry ${money.format(t.entry_price)}</span></div>
        <div class="cushion-track"><div class="cushion-fill" style="width:${progress.toFixed(1)}%"></div></div>
      </div>
      <div class="muted tiny">Opened ${timeOnly(t.entry_ts)} · ${esc(t.trade_id || "")}</div>
    </div>`;
}

function renderLosses() {
  const l = DATA.losses;
  $("lossSummary").innerHTML = `
    <div class="loss-summary">
      <div><span>Losses</span><strong>${l.count}</strong></div>
      <div><span>Total</span><strong class="red">${fmtMoney(l.total)}</strong></div>
      <div><span>Average</span><strong class="red">${fmtMoney(l.avg)}</strong></div>
      <div><span>Worst streak</span><strong>${l.streak}</strong></div>
    </div>`;
  if (!l.worst.length) {
    $("worstLosses").innerHTML = `<div class="muted tiny">No losing trades yet.</div>`;
    return;
  }
  $("worstLosses").innerHTML = `<div class="worst">${l.worst.map(t => `
    <div class="worst-row"><span>${esc(dateOnly(t.date))} · ${esc(t.reason)}</span><strong class="red">${fmtMoney(t.pnl)}</strong></div>`).join("")}</div>`;
}

function renderHealth() {
  const h = DATA.health;
  const badge = $("healthBadge");
  badge.textContent = DATA.meta.demo ? "Demo" : h.stale_data ? "Stale" : h.daily_loss_halt ? "Halted" : h.market_open === true ? "Live" : h.market_open === false ? "Closed" : "Idle";
  badge.className = "badge " + (DATA.meta.demo ? "amber" : h.stale_data || h.daily_loss_halt ? "red" : h.market_open === true ? "green" : "amber");
  const rows = [
    ["Last run", etDateTime(h.last_run)],
    ["Last bar", etTime(h.last_bar)],
    ["Data age", h.data_age_min == null ? "—" : `${fmtNum(h.data_age_min, 0)} min`],
    ["Trades today", h.trades_today == null ? "—" : `${h.trades_today} / ${DATA.strategy.max_trades_per_day || 1}`],
    ["Last action", h.last_action ? `${h.last_action} · ${h.last_reason || ""}` : "—"],
    ["Daily loss halt", h.daily_loss_halt ? "Yes" : "No"],
  ];
  $("health").innerHTML = rows.map(([k, v]) => `<div class="health-row"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join("") +
    (h.stale_data ? `<div class="alert warn">The last bar is older than the configured limit. Entries are blocked until fresh data arrives.</div>` : "") +
    (h.daily_loss_halt ? `<div class="alert">Daily loss limit hit. No new entries today; the position is flattened.</div>` : "") +
    (h.error ? `<div class="alert">${esc(h.error)}</div>` : "");
}

function svgLine(points, opts) {
  const W = 820, H = 280, P = { t: 18, r: 18, b: 30, l: 62 };
  const vals = points.map(p => Number(p[opts.y]));
  let min = Math.min(...vals), max = Math.max(...vals);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.12; min -= pad; max += pad;
  const x = i => P.l + (i / (points.length - 1)) * (W - P.l - P.r);
  const y = v => P.t + (1 - (v - min) / (max - min)) * (H - P.t - P.b);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${y(Number(p[opts.y])).toFixed(2)}`).join(" ");
  const area = `${path} L${x(points.length - 1).toFixed(2)},${H - P.b} L${x(0).toFixed(2)},${H - P.b} Z`;
  const gid = "g" + opts.id;
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const gy = P.t + (i / 4) * (H - P.t - P.b);
    const gv = max - (i / 4) * (max - min);
    grid += `<line x1="${P.l}" y1="${gy}" x2="${W - P.r}" y2="${gy}" stroke="#1f2a3a" stroke-width="1"/>`;
    grid += `<text x="${P.l - 9}" y="${gy + 4}" fill="#61708a" font-size="10" text-anchor="end">${opts.money ? money0.format(gv) : gv.toFixed(1)}</text>`;
  }
  const labelIdx = [0, Math.floor((points.length - 1) / 2), points.length - 1];
  const labels = [...new Set(labelIdx)].map(i => `<text x="${x(i)}" y="${H - 8}" fill="#61708a" font-size="10" text-anchor="${i === 0 ? "start" : i === points.length - 1 ? "end" : "middle"}">${esc(dateOnly(points[i][opts.x]))}</text>`).join("");
  const dots = points.map((p, i) => `<circle cx="${x(i)}" cy="${y(Number(p[opts.y]))}" r="7" fill="transparent"><title>${esc(dateOnly(p[opts.x]))}: ${opts.money ? money.format(Number(p[opts.y])) : p[opts.y]}</title></circle>`).join("");
  const last = points[points.length - 1];
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label)}">
    <defs><linearGradient id="${gid}" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="${opts.color}" stop-opacity=".28"/>
      <stop offset="100%" stop-color="${opts.color}" stop-opacity="0"/>
    </linearGradient></defs>
    ${grid}
    <path d="${area}" fill="url(#${gid})"/>
    <path d="${path}" fill="none" stroke="${opts.color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${x(points.length - 1)}" cy="${y(Number(last[opts.y]))}" r="4" fill="${opts.color}" stroke="#0a0e16" stroke-width="2"/>
    ${dots}${labels}
  </svg>`;
}

function svgBars(points, opts) {
  const W = 820, H = 280, P = { t: 18, r: 18, b: 30, l: 62 };
  const vals = points.map(p => Number(p[opts.y]));
  const maxAbs = Math.max(...vals.map(Math.abs), 1);
  const x = i => P.l + (i + 0.15) * ((W - P.l - P.r) / points.length);
  const bw = Math.max(2, ((W - P.l - P.r) / points.length) * 0.7);
  const y = v => P.t + (1 - (v + maxAbs) / (2 * maxAbs)) * (H - P.t - P.b);
  const zero = y(0);
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const gy = P.t + (i / 4) * (H - P.t - P.b);
    const gv = maxAbs - (i / 4) * (2 * maxAbs);
    grid += `<line x1="${P.l}" y1="${gy}" x2="${W - P.r}" y2="${gy}" stroke="#1f2a3a" stroke-width="1"/>`;
    grid += `<text x="${P.l - 9}" y="${gy + 4}" fill="#61708a" font-size="10" text-anchor="end">${money0.format(gv)}</text>`;
  }
  const bars = points.map((p, i) => {
    const v = Number(p[opts.y]);
    const top = v >= 0 ? y(v) : zero;
    const h = Math.abs(y(v) - zero);
    const color = v >= 0 ? "#34d399" : "#f87171";
    return `<rect x="${x(i)}" y="${top}" width="${bw}" height="${Math.max(h, 1)}" rx="2" fill="${color}" opacity=".9"><title>${esc(dateOnly(p[opts.x]))}: ${fmtMoney(v)}</title></rect>`;
  }).join("");
  const step = Math.max(1, Math.ceil(points.length / 6));
  const labels = points.map((p, i) => i % step === 0 || i === points.length - 1 ? `<text x="${x(i) + bw / 2}" y="${H - 8}" fill="#61708a" font-size="10" text-anchor="middle">${esc(dateOnly(p[opts.x]))}</text>` : "").join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label)}">
    ${grid}<line x1="${P.l}" y1="${zero}" x2="${W - P.r}" y2="${zero}" stroke="#334155" stroke-width="1"/>${bars}${labels}
  </svg>`;
}

function renderCharts() {
  const eq = DATA.equity_curve;
  if (eq.length < 2) {
    $("equityChart").innerHTML = empty("No equity curve yet", "It appears after the first closed trade.");
  } else {
    $("equityChart").innerHTML = svgLine(eq, { x: "date", y: "equity", color: "#60a5fa", money: true, label: "Equity curve", id: "eq" });
  }
  const last = eq.length ? eq[eq.length - 1].equity : DATA.account.starting_equity;
  $("equityMeta").textContent = `${money0.format(last)} · ${DATA.metrics.trades || 0} closed`;
  const daily = DATA.daily_pnl;
  if (!daily.length) {
    $("dailyChart").innerHTML = empty("No daily P&L yet", "One bar per session with a closed trade.");
  } else {
    $("dailyChart").innerHTML = svgBars(daily, { x: "date", y: "pnl", label: "Daily P&L", id: "daily" });
  }
  $("dailyMeta").textContent = `${daily.length} trade days`;
}

function renderBands() {
  const h = DATA.health;
  const rows = [
    ["Open", h.open == null ? "—" : money.format(h.open)],
    ["Upper band", h.upper == null ? "—" : money.format(h.upper)],
    ["Lower band", h.lower == null ? "—" : money.format(h.lower)],
    ["VWAP", h.vwap == null ? "—" : money.format(h.vwap)],
    ["Last price", h.price == null ? "—" : money.format(h.price)],
    ["Sigma", h.sigma == null ? "—" : fmtPct(h.sigma, 3, false)],
  ];
  $("bandGrid").innerHTML = rows.map(([k, v]) => `<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join("");
  $("bandNote").textContent = h.prior_close == null ? "Waiting for a completed prior session." : `Prior close ${money.format(h.prior_close)} · bands are gap-aware.`;
}

function renderNotes() {
  $("notes").innerHTML = DATA.notes.map(n => `<li>${esc(n)}</li>`).join("");
}

function tradeRow(t) {
  return `<tr>
    <td>${esc(dateOnly(t.date))}</td>
    <td class="dim">${esc(timeOnly(t.entry_time))} → ${esc(timeOnly(t.exit_time))}</td>
    <td>${esc(t.side)}</td>
    <td class="num">${t.qty}</td>
    <td class="num">${money.format(t.entry)}</td>
    <td class="num">${money.format(t.exit)}</td>
    <td class="num ${tone(t.pnl)}">${fmtMoney(t.pnl)}</td>
    <td class="num ${tone(t.r)}">${fmtR(t.r)}</td>
    <td class="reason">${esc(t.reason)}</td>
    <td class="num dim">${fmtNum(t.hold_minutes, 0)}m</td>
  </tr>`;
}

function renderTrades() {
  const rows = DATA.trades;
  if (!rows.length) {
    $("tradeTable").innerHTML = empty("No closed trades yet", "The bot only trades when price leaves the noise area.");
    return;
  }
  $("tradeTable").innerHTML = `<table><thead><tr><th>Date</th><th>Time</th><th>Side</th><th class="num">Qty</th><th class="num">Entry</th><th class="num">Exit</th><th class="num">P&amp;L</th><th class="num">R</th><th>Reason</th><th class="num">Hold</th></tr></thead><tbody id="tradeBody"></tbody></table>`;
  applyTradeFilter();
}

let tradeFilter = "all";
function applyTradeFilter() {
  const body = $("tradeBody");
  if (!body) return;
  const q = ($("tradeSearch")?.value || "").toLowerCase().trim();
  const rows = DATA.trades.filter(t => {
    if (tradeFilter === "wins" && !(t.pnl > 0)) return false;
    if (tradeFilter === "losses" && !(t.pnl < 0)) return false;
    if (!q) return true;
    return `${t.date} ${t.reason} ${t.trade_id}`.toLowerCase().includes(q);
  });
  body.innerHTML = rows.length ? rows.map(tradeRow).join("") : `<tr><td colspan="10" class="dim" style="text-align:center;padding:24px">No trades match this filter.</td></tr>`;
}

function renderLossAnalysis() {
  const l = DATA.losses;
  const m = DATA.metrics;
  $("lossKpis").innerHTML = `
    <div class="kpi"><span>Losing trades</span><strong>${l.count}</strong><small>${m.trades ? fmtPct(l.count / m.trades, 1, false) : "0%"} of closed</small></div>
    <div class="kpi"><span>Total lost</span><strong class="red">${fmtMoney(l.total)}</strong><small>gross losses</small></div>
    <div class="kpi"><span>Average loss</span><strong class="red">${fmtMoney(l.avg)}</strong><small>per losing trade</small></div>
    <div class="kpi"><span>Worst trade</span><strong class="red">${fmtMoney(l.largest)}</strong><small>single loss</small></div>`;
  if (!l.worst.length) {
    $("lossTable").innerHTML = empty("No losing trades", "That is a good problem to have — or a very short sample.");
  } else {
    $("lossTable").innerHTML = `<table><thead><tr><th>Date</th><th>Reason</th><th class="num">P&amp;L</th><th class="num">R</th><th class="num">Hold</th></tr></thead><tbody>${l.worst.map(t => `<tr><td>${esc(dateOnly(t.date))}</td><td class="reason">${esc(t.reason)}</td><td class="num red">${fmtMoney(t.pnl)}</td><td class="num red">${fmtR(t.r)}</td><td class="num dim">${fmtNum(t.hold_minutes, 0)}m</td></tr>`).join("")}</tbody></table>`;
  }
  const entries = Object.entries(l.by_reason).sort((a, b) => a[1] - b[1]);
  $("lossReasons").innerHTML = entries.length ? `<div class="worst">${entries.map(([reason, pnl]) => `<div class="worst-row"><span>${esc(reason)}</span><strong class="red">${fmtMoney(pnl)}</strong></div>`).join("")}</div>` : `<div class="muted tiny">No losses by reason yet.</div>`;
}

function renderRisk() {
  const s = DATA.strategy;
  const h = DATA.health;
  const rows = [
    ["Symbol", DATA.bot.symbol],
    ["Direction", "Long only"],
    ["Risk / trade", fmtPct(s.risk_per_trade, 2, false)],
    ["Max position", fmtPct(s.max_position_pct || 1.0, 0, false)],
    ["Max trades / day", String(s.max_trades_per_day)],
    ["Daily loss halt", fmtPct(s.max_daily_loss_pct || 0.01, 2, false)],
    ["Band lookback", `${s.band_lookback_sessions} sessions`],
    ["Band multiplier", fmtNum(s.band_mult, 2)],
    ["Flat time", s.flat_time || "15:45"],
    ["Data feed", "IEX (free)"],
    ["Execution", "Paper only"],
    ["Kill switch", h.daily_loss_halt ? "TRIPPED" : "Armed"],
  ];
  $("settings").innerHTML = rows.map(([k, v]) => `<div class="setting"><span>${esc(k)}</span><strong class="${k === "Kill switch" && h.daily_loss_halt ? "red" : ""}">${esc(v)}</strong></div>`).join("");
  const alerts = [];
  if (h.daily_loss_halt) alerts.push(`<div class="alert">Daily loss halt is active. The sleeve will not enter new trades today.</div>`);
  if (h.stale_data) alerts.push(`<div class="alert warn">Data is stale. Entries are blocked.</div>`);
  if (h.error) alerts.push(`<div class="alert">${esc(h.error)}</div>`);
  if (!alerts.length) alerts.push(`<div class="muted tiny">No risk alerts. The sleeve is inside its limits.</div>`);
  $("riskAlerts").innerHTML = alerts.join("");
}

function render() {
  renderHeader();
  renderHero();
  renderOpen();
  renderLosses();
  renderHealth();
  renderCharts();
  renderBands();
  renderNotes();
  renderTrades();
  renderLossAnalysis();
  renderRisk();
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = $("tab-" + btn.dataset.tab);
      if (panel) panel.classList.add("active");
    });
  });
  document.querySelectorAll(".chip[data-filter]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".chip[data-filter]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      tradeFilter = btn.dataset.filter;
      applyTradeFilter();
    });
  });
  const search = $("tradeSearch");
  if (search) search.addEventListener("input", applyTradeFilter);
}

render();
if (DATA.meta.live) {
  setTimeout(() => location.reload(), 30000);
}
</script>
</body>
</html>
"""
