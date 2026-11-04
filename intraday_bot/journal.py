"""Append-only journal and dashboard metrics for the intraday sleeve."""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def record(path: str | Path, event: str, **fields: Any) -> None:
    line = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str) + "\n")


def load(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    events: list[dict] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "event" in obj:
            events.append(obj)
    return events


def closed_trades(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        if e.get("event") != "trade_closed":
            continue
        if not isinstance(e.get("pnl"), (int, float)):
            continue
        out.append(e)
    return sorted(out, key=lambda e: str(e.get("exit_ts") or e.get("ts") or ""))


def open_trade(events: list[dict]) -> dict | None:
    """Last entry that has no matching close."""
    closed_ids = {str(e.get("trade_id")) for e in closed_trades(events) if e.get("trade_id")}
    entries = [e for e in events if e.get("event") == "entry"]
    for e in reversed(entries):
        tid = str(e.get("trade_id") or "")
        if tid and tid not in closed_ids:
            return e
    return None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def metrics(events: list[dict], starting_equity: float = 100_000.0) -> dict:
    trades = closed_trades(events)
    pnls = [_num(t.get("pnl")) for t in trades]
    rs = [_num(t.get("r_multiple")) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    net = sum(pnls)
    dates = [str(t.get("date") or str(t.get("exit_ts", ""))[:10]) for t in trades]
    if pnls:
        daily = pd.Series(pnls, index=pd.Index(dates, name="date")).groupby(level=0).sum()
        equity = starting_equity + daily.cumsum()
        equity = pd.concat([pd.Series([starting_equity], index=["start"]), equity])
    else:
        daily = pd.Series(dtype=float)
        equity = pd.Series([starting_equity], index=["start"])
    daily_returns = equity.pct_change().dropna()
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if len(daily_returns) > 1 and daily_returns.std() > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series([0.0])
    streak = 0
    max_streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    by_reason: dict[str, float] = {}
    for t in trades:
        if _num(t.get("pnl")) < 0:
            reason = str(t.get("reason", "unknown"))
            by_reason[reason] = by_reason.get(reason, 0.0) + _num(t.get("pnl"))
    return {
        "has_data": bool(trades),
        "trades": len(trades),
        "sessions": int(len(daily)),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "net_pnl": net,
        "total_return_pct": (net / starting_equity) if starting_equity else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "expectancy": float(np.mean(pnls)) if pnls else 0.0,
        "avg_r": float(np.mean(rs)) if rs else 0.0,
        "total_r": float(np.sum(rs)) if rs else 0.0,
        "max_drawdown_pct": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe": sharpe,
        "best_trade": max(pnls) if pnls else 0.0,
        "worst_trade": min(pnls) if pnls else 0.0,
        "avg_hold_minutes": float(np.mean([_num(t.get("hold_minutes")) for t in trades])) if trades else 0.0,
        "loss_count": len(losses),
        "loss_total": sum(losses),
        "loss_avg": float(np.mean(losses)) if losses else 0.0,
        "loss_largest": min(losses) if losses else 0.0,
        "loss_streak": max_streak,
        "loss_by_reason": by_reason,
        "small_sample": len(trades) < 30,
    }


def daily_pnl(events: list[dict]) -> list[dict]:
    trades = closed_trades(events)
    if not trades:
        return []
    frame = pd.DataFrame(
        {
            "date": [str(t.get("date") or str(t.get("exit_ts", ""))[:10]) for t in trades],
            "pnl": [_num(t.get("pnl")) for t in trades],
        }
    )
    grouped = frame.groupby("date", sort=True)["pnl"].agg(["sum", "count"]).reset_index()
    return [
        {"date": str(row["date"]), "pnl": float(row["sum"]), "trades": int(row["count"])}
        for _, row in grouped.iterrows()
    ]


def equity_curve(events: list[dict], starting_equity: float = 100_000.0) -> list[dict]:
    trades = closed_trades(events)
    if not trades:
        return []
    out = [{"date": "start", "equity": starting_equity, "pnl": 0.0}]
    equity = starting_equity
    for t in trades:
        equity += _num(t.get("pnl"))
        out.append(
            {
                "date": str(t.get("date") or str(t.get("exit_ts", ""))[:10]),
                "equity": equity,
                "pnl": _num(t.get("pnl")),
            }
        )
    return out


def trade_rows(events: list[dict]) -> list[dict]:
    rows = []
    for t in closed_trades(events):
        rows.append(
            {
                "trade_id": t.get("trade_id", ""),
                "date": str(t.get("date") or str(t.get("exit_ts", ""))[:10]),
                "entry_time": str(t.get("entry_ts", "")),
                "exit_time": str(t.get("exit_ts", "")),
                "side": str(t.get("side", "long")),
                "qty": int(_num(t.get("qty"))),
                "entry": _num(t.get("entry_price")),
                "exit": _num(t.get("exit_price")),
                "stop": _num(t.get("stop")),
                "pnl": _num(t.get("pnl")),
                "ret_pct": _num(t.get("ret")),
                "r": _num(t.get("r_multiple")),
                "reason": str(t.get("reason", "")),
                "hold_minutes": _num(t.get("hold_minutes")),
                "mae": _num(t.get("mae")),
                "mfe": _num(t.get("mfe")),
            }
        )
    return rows
