"""Unified read-only market viewer.

This is the canonical dashboard for the repo. It replaces the two older
dashboards:

- the Alpaca-account Streamlit page in ``dashboard.py``
- the SPY-only intraday page in ``intraday_bot/dashboard.py``

It is deliberately boring: no API keys, no broker imports, no order endpoints,
no external CDN, and no writes. It reads local OHLCV CSVs and the two existing
journals, then serves or writes one self-contained HTML page.
"""
from __future__ import annotations

import argparse
import html
import json
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import pandas as pd

from market_data import CsvMarketDataProvider, MarketDataProvider

DEFAULT_ROOTS = ("data", "intraday")
DEFAULT_SWING_JOURNAL = "journal/trades.jsonl"
DEFAULT_INTRADAY_JOURNAL = "intraday/journal.jsonl"
DEFAULT_INTRADAY_STATE = "intraday/state.json"


def _read_json(path: str | Path) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _swing_summary(path: str | Path) -> dict:
    try:
        from trader.journal import load, summarize

        events = load(path)
        stats = summarize(events)
        return {"journal": str(path), "events": len(events), **stats}
    except Exception as exc:  # the viewer must not crash because one sleeve is missing
        return {"journal": str(path), "error": str(exc)}


def _intraday_summary(path: str | Path, state_path: str | Path) -> dict:
    try:
        from intraday_bot.journal import load, metrics

        state = _read_json(state_path)
        starting_equity = float(state.get("starting_equity", 100_000.0) or 100_000.0)
        events = load(path)
        stats = metrics(events, starting_equity)
        return {
            "journal": str(path),
            "events": len(events),
            "starting_equity": starting_equity,
            "trades": stats["trades"],
            "net_pnl": stats["net_pnl"],
            "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "max_drawdown_pct": stats["max_drawdown_pct"],
            "small_sample": stats["small_sample"],
        }
    except Exception as exc:
        return {"journal": str(path), "error": str(exc)}


def _bar_interval(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "unknown"
    delta = pd.Series(index).diff().dropna().median()
    seconds = float(delta.total_seconds())
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86_400:
        return f"{seconds / 3600:.0f}h"
    return f"{seconds / 86_400:.0f}d"


def _asset_payload(provider: MarketDataProvider, symbol: str, max_bars: int) -> dict:
    try:
        frame = provider.get_bars(symbol)
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}

    if frame.empty:
        return {"symbol": symbol, "error": "no bars"}

    last = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else last
    change = float(last["close"] - previous["close"])
    change_pct = change / float(previous["close"]) if float(previous["close"]) else 0.0
    chart = frame.tail(max_bars)

    bars = []
    for timestamp, row in chart.iterrows():
        bars.append(
            {
                "t": timestamp.isoformat(),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": float(row["volume"]),
            }
        )

    return {
        "symbol": symbol,
        "source": frame.attrs.get("source", ""),
        "rows": int(len(frame)),
        "first": frame.index[0].isoformat(),
        "last": frame.index[-1].isoformat(),
        "bar_interval": _bar_interval(frame.index),
        "last_close": float(last["close"]),
        "change": change,
        "change_pct": change_pct,
        "period_high": float(frame["high"].max()),
        "period_low": float(frame["low"].min()),
        "period_volume": float(frame["volume"].sum()),
        "invalid_ohlc": int(frame.attrs.get("invalid_ohlc", 0)),
        "bars": bars,
    }


def build_payload(
    provider: MarketDataProvider,
    max_bars: int = 180,
    swing_journal: str | Path = DEFAULT_SWING_JOURNAL,
    intraday_journal: str | Path = DEFAULT_INTRADAY_JOURNAL,
    intraday_state: str | Path = DEFAULT_INTRADAY_STATE,
) -> dict:
    assets = []
    errors = []
    for symbol in provider.list_symbols():
        asset = _asset_payload(provider, symbol, max_bars=max_bars)
        assets.append(asset)
        if asset.get("error"):
            errors.append({"symbol": symbol, "error": asset["error"]})

    total_rows = sum(int(asset.get("rows", 0)) for asset in assets)
    total_invalid = sum(int(asset.get("invalid_ohlc", 0)) for asset in assets)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read-only",
        "provider": provider.name,
        "assets": assets,
        "sleeves": {
            "swing": _swing_summary(swing_journal),
            "intraday": _intraday_summary(intraday_journal, intraday_state),
        },
        "data_health": {
            "assets": len(assets),
            "rows": total_rows,
            "invalid_ohlc": total_invalid,
            "errors": errors,
        },
        "security": {
            "read_only": True,
            "api_keys": False,
            "broker_orders": False,
            "external_network": False,
            "writes": False,
            "bind": "127.0.0.1",
        },
    }


def _svg_candles(bars: list[dict], width: int = 640, height: int = 230) -> str:
    if not bars:
        return '<div class="empty">No bars.</div>'

    pad = 12
    low = min(bar["l"] for bar in bars)
    high = max(bar["h"] for bar in bars)
    span = max(high - low, 1e-9)
    step = (width - 2 * pad) / max(len(bars), 1)
    body_width = max(1.0, step * 0.58)

    def y(price: float) -> float:
        return pad + (high - price) / span * (height - 2 * pad)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="candlestick chart">',
    ]
    for fraction in (0.25, 0.5, 0.75):
        grid_y = pad + fraction * (height - 2 * pad)
        parts.append(
            f'<line x1="{pad}" y1="{grid_y:.2f}" x2="{width - pad}" y2="{grid_y:.2f}" class="grid"/>'
        )

    for index, bar in enumerate(bars):
        x = pad + index * step + step / 2
        color = "up" if bar["c"] >= bar["o"] else "down"
        top = y(max(bar["o"], bar["c"]))
        bottom = y(min(bar["o"], bar["c"]))
        body_height = max(1.0, bottom - top)
        title = html.escape(f'{bar["t"]}  O {bar["o"]:.4f}  H {bar["h"]:.4f}  L {bar["l"]:.4f}  C {bar["c"]:.4f}')
        parts.append(f"<g><title>{title}</title>")
        parts.append(
            f'<line x1="{x:.2f}" y1="{y(bar["h"]):.2f}" x2="{x:.2f}" y2="{y(bar["l"]):.2f}" class="{color}"/>'
        )
        parts.append(
            f'<rect x="{x - body_width / 2:.2f}" y="{top:.2f}" width="{body_width:.2f}" '
            f'height="{body_height:.2f}" class="{color}"/>'
        )
        parts.append("</g>")

    parts.append("</svg>")
    return "".join(parts)


def _asset_card(asset: dict) -> str:
    symbol = html.escape(str(asset.get("symbol", "?")))
    if asset.get("error"):
        return (
            f'<article class="card asset"><div class="asset-head"><h2>{symbol}</h2>'
            f'<span class="badge bad">ERROR</span></div>'
            f'<div class="empty">{html.escape(str(asset["error"]))}</div></article>'
        )

    change = float(asset["change"])
    change_pct = float(asset["change_pct"])
    direction = "up" if change >= 0 else "down"
    invalid = int(asset["invalid_ohlc"])
    invalid_badge = (
        f'<span class="badge warn">{invalid} OHLC violations</span>'
        if invalid
        else '<span class="badge ok">OHLC clean</span>'
    )
    source = html.escape(str(asset.get("source", "")))
    return f"""
    <article class="card asset">
      <div class="asset-head">
        <div>
          <h2>{symbol}</h2>
          <p class="muted tiny">{html.escape(str(asset["bar_interval"]))} bars · {int(asset["rows"]):,} rows</p>
        </div>
        <div class="asset-price">
          <strong>{float(asset["last_close"]):,.4f}</strong>
          <span class="{direction}">{change:+,.4f} ({change_pct:+.2%})</span>
        </div>
      </div>
      <div class="chart">{_svg_candles(asset["bars"])}</div>
      <div class="metrics">
        <div><span>Period high</span><strong>{float(asset["period_high"]):,.4f}</strong></div>
        <div><span>Period low</span><strong>{float(asset["period_low"]):,.4f}</strong></div>
        <div><span>First bar</span><strong>{html.escape(str(asset["first"]))}</strong></div>
        <div><span>Last bar</span><strong>{html.escape(str(asset["last"]))}</strong></div>
      </div>
      <div class="asset-foot">
        {invalid_badge}
        <span class="muted tiny">{source}</span>
      </div>
    </article>
    """


def _sleeve_card(title: str, stats: dict) -> str:
    if stats.get("error"):
        body = f'<div class="empty">{html.escape(str(stats["error"]))}</div>'
    elif title == "Swing sleeve":
        body = f"""
        <div class="metrics">
          <div><span>Closed trades</span><strong>{int(stats.get("trades", 0))}</strong></div>
          <div><span>Win rate</span><strong>{float(stats.get("win_rate", 0.0)):.1%}</strong></div>
          <div><span>Avg return</span><strong>{float(stats.get("avg_return", 0.0)):+.2%}</strong></div>
          <div><span>Skips</span><strong>{int(stats.get("skips", 0))}</strong></div>
        </div>
        """
    else:
        profit_factor = stats.get("profit_factor")
        pf = "—" if profit_factor is None else f"{float(profit_factor):.2f}"
        body = f"""
        <div class="metrics">
          <div><span>Closed trades</span><strong>{int(stats.get("trades", 0))}</strong></div>
          <div><span>Net P&amp;L</span><strong>{float(stats.get("net_pnl", 0.0)):+,.2f}</strong></div>
          <div><span>Win rate</span><strong>{float(stats.get("win_rate", 0.0)):.1%}</strong></div>
          <div><span>Profit factor</span><strong>{pf}</strong></div>
          <div><span>Max drawdown</span><strong>{float(stats.get("max_drawdown_pct", 0.0)):.2%}</strong></div>
        </div>
        """
    warning = (
        '<p class="warn-text">Fewer than 30 closed trades. Treat these numbers as noise.</p>'
        if stats.get("small_sample")
        else ""
    )
    return f"""
    <article class="card sleeve">
      <div class="asset-head"><h2>{html.escape(title)}</h2><span class="badge">READ-ONLY</span></div>
      {body}
      {warning}
      <p class="muted tiny">{html.escape(str(stats.get("journal", "")))}</p>
    </article>
    """


def render_html(payload: dict, refresh_seconds: int | None = None) -> str:
    refresh = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds
        else ""
    )
    cards = "".join(_asset_card(asset) for asset in payload.get("assets", []))
    if not cards:
        cards = '<div class="empty">No OHLCV CSVs found. Add one under data/ or intraday/.</div>'

    sleeves = payload.get("sleeves", {})
    sleeve_cards = _sleeve_card("Swing sleeve", sleeves.get("swing", {})) + _sleeve_card(
        "Intraday sleeve", sleeves.get("intraday", {})
    )
    health = payload.get("data_health", {})
    errors = health.get("errors", [])
    error_text = (
        " · ".join(f'{item["symbol"]}: {item["error"]}' for item in errors)
        if errors
        else "none"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}
<title>Market Viewer</title>
<style>
:root {{
  --bg: #0a0e16; --panel: #111827; --panel-2: #0d1422; --line: #1f2a3a;
  --line-soft: #172233; --text: #e6edf7; --muted: #8b9bb4; --muted-2: #61708a;
  --green: #34d399; --red: #f87171; --amber: #fbbf24; --blue: #60a5fa;
  --radius: 16px;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background:
    radial-gradient(1100px 500px at 15% -10%, rgba(96, 165, 250, .10), transparent 60%),
    radial-gradient(900px 420px at 100% 0%, rgba(167, 139, 250, .08), transparent 55%),
    var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px; line-height: 1.5; min-height: 100vh;
}}
.wrap {{ max-width: 1240px; margin: 0 auto; padding: 24px 22px 48px; }}
.topbar {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 8px 0 22px; }}
.brand h1 {{ font-size: 22px; margin: 0; letter-spacing: -.02em; }}
.brand p {{ margin: 4px 0 0; color: var(--muted); font-size: 12px; }}
.badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{
  display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px;
  border-radius: 999px; border: 1px solid var(--line); background: var(--panel-2);
  color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .04em;
}}
.badge.ok {{ color: var(--green); border-color: rgba(52, 211, 153, .28); }}
.badge.bad {{ color: var(--red); border-color: rgba(248, 113, 113, .28); }}
.badge.warn {{ color: var(--amber); border-color: rgba(251, 191, 36, .25); }}
.summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }}
.kpi {{ border: 1px solid var(--line); background: var(--panel); border-radius: 14px; padding: 14px 15px; }}
.kpi span {{ display: block; color: var(--muted); font-size: 11px; margin-bottom: 6px; }}
.kpi strong {{ display: block; font-size: 19px; letter-spacing: -.02em; }}
.kpi small {{ display: block; color: var(--muted-2); font-size: 11px; margin-top: 4px; }}
.assets {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 18px; }}
.card {{ border: 1px solid var(--line); background: var(--panel); border-radius: var(--radius); padding: 18px; }}
.asset-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }}
.asset-head h2 {{ font-size: 16px; margin: 0; letter-spacing: -.01em; }}
.asset-price {{ text-align: right; }}
.asset-price strong {{ display: block; font-size: 20px; letter-spacing: -.02em; }}
.asset-price span {{ font-size: 12px; font-weight: 700; }}
.chart {{ min-height: 230px; }}
.chart svg {{ width: 100%; height: auto; display: block; }}
.chart .grid {{ stroke: var(--line-soft); stroke-width: 1; }}
.chart .up {{ stroke: var(--green); fill: var(--green); }}
.chart .down {{ stroke: var(--red); fill: var(--red); }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 14px; }}
.metrics div {{ background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 10px; padding: 9px 10px; }}
.metrics span {{ display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }}
.metrics strong {{ display: block; font-size: 12px; margin-top: 3px; word-break: break-word; }}
.asset-foot {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 12px; flex-wrap: wrap; }}
.sleeves {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 18px; }}
.sleeve .metrics {{ grid-template-columns: repeat(4, 1fr); }}
.empty {{ min-height: 160px; display: grid; place-items: center; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 12px; padding: 22px; }}
.muted {{ color: var(--muted); }}
.tiny {{ font-size: 11px; }}
.up {{ color: var(--green); }}
.down {{ color: var(--red); }}
.warn-text {{ color: var(--amber); font-size: 12px; margin: 12px 0 0; }}
.footer {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; color: var(--muted-2); font-size: 11px; padding-top: 22px; border-top: 1px solid var(--line-soft); }}
@media (max-width: 980px) {{
  .summary, .assets, .sleeves {{ grid-template-columns: 1fr 1fr; }}
  .metrics, .sleeve .metrics {{ grid-template-columns: 1fr 1fr; }}
}}
@media (max-width: 620px) {{
  .wrap {{ padding: 16px 14px 36px; }}
  .summary, .assets, .sleeves {{ grid-template-columns: 1fr; }}
  .topbar {{ flex-direction: column; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <header class="topbar">
    <div class="brand">
      <h1>Market Viewer</h1>
      <p>Read-only multi-asset viewer · local CSVs + paper journals · no order path</p>
    </div>
    <div class="badges">
      <span class="badge ok">READ-ONLY</span>
      <span class="badge ok">NO API KEYS</span>
      <span class="badge">LOCALHOST ONLY</span>
    </div>
  </header>

  <section class="summary">
    <div class="kpi"><span>Assets</span><strong>{int(health.get("assets", 0))}</strong><small>{html.escape(str(payload.get("provider", "?")))} provider</small></div>
    <div class="kpi"><span>Bars loaded</span><strong>{int(health.get("rows", 0)):,}</strong><small>all assets</small></div>
    <div class="kpi"><span>OHLC violations</span><strong>{int(health.get("invalid_ohlc", 0)):,}</strong><small>flagged, not repaired</small></div>
    <div class="kpi"><span>Generated</span><strong>{html.escape(str(payload.get("generated_at", ""))[:19])}</strong><small>UTC</small></div>
  </section>

  <section class="assets">
    {cards}
  </section>

  <section class="sleeves">
    {sleeve_cards}
  </section>

    <footer class="footer">
      <span>No API keys are read, stored, or rendered. No broker or order endpoints are imported. GET-only {html.escape(str(payload.get("security", {}).get("bind", "127.0.0.1")))}</span>
      <span>Data errors: {html.escape(error_text)}</span>
    </footer>
</div>
</body>
</html>
"""


def write_static(payload: dict, path: str | Path, refresh_seconds: int | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(payload, refresh_seconds=refresh_seconds), encoding="utf-8")
    return target


def make_server(
    payload_factory: Callable[[], dict],
    port: int = 8770,
    refresh_seconds: int = 60,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                payload = payload_factory()
                self._send(
                    200,
                    render_html(payload, refresh_seconds=refresh_seconds).encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/api/data":
                payload = payload_factory()
                self._send(
                    200,
                    json.dumps(payload, default=str).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            elif self.path == "/health":
                self._send(
                    200,
                    b'{"ok":true,"read_only":true,"api_keys":false}',
                    "application/json; charset=utf-8",
                )
            else:
                self._send(404, b'{"error":"not found"}', "application/json; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            self._send(405, b'{"error":"read-only"}', "application/json; charset=utf-8")

        def do_PUT(self) -> None:  # noqa: N802
            self._send(405, b'{"error":"read-only"}', "application/json; charset=utf-8")

        def do_DELETE(self) -> None:  # noqa: N802
            self._send(405, b'{"error":"read-only"}', "application/json; charset=utf-8")

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(
    payload_factory: Callable[[], dict],
    port: int = 8770,
    open_browser: bool = False,
) -> None:
    server = make_server(payload_factory, port=port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Market viewer: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def default_provider() -> CsvMarketDataProvider:
    return CsvMarketDataProvider(roots=DEFAULT_ROOTS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only multi-asset market viewer")
    parser.add_argument("--data-dir", action="append", help="directory with OHLCV CSVs (repeatable)")
    parser.add_argument("--swing-journal", default=DEFAULT_SWING_JOURNAL)
    parser.add_argument("--intraday-journal", default=DEFAULT_INTRADAY_JOURNAL)
    parser.add_argument("--intraday-state", default=DEFAULT_INTRADAY_STATE)
    parser.add_argument("--max-bars", type=int, default=180)
    parser.add_argument("--out", default="market_viewer.html")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)

    roots = tuple(args.data_dir) if args.data_dir else DEFAULT_ROOTS
    provider = CsvMarketDataProvider(roots=roots)

    def payload() -> dict:
        return build_payload(
            provider,
            max_bars=args.max_bars,
            swing_journal=args.swing_journal,
            intraday_journal=args.intraday_journal,
            intraday_state=args.intraday_state,
        )

    if args.serve:
        serve(payload, port=args.port, open_browser=args.open)
        return 0

    target = write_static(payload(), args.out)
    print(f"Wrote {target}")
    if args.open:
        webbrowser.open(target.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
