"""Append-only trade journal, one JSON object per line.

Records what the agent saw, chose, and why, so results can be audited later.
Read-only consumers (a dashboard, a report) use `load` and `summarize`.
Bad lines are skipped rather than crashing the reader, so one corrupt line
cannot hide the rest of the history.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record(path: str | Path, event: str, **fields: Any) -> None:
    """Append one event. Creates the file and its folder if needed."""
    line = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str) + "\n")


def load(path: str | Path) -> list[dict]:
    """Read every valid event. Missing file means an empty history."""
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


def summarize(events: list[dict]) -> dict:
    """Win rate and exit reasons, computed from closed trades only.

    A closed trade is an event with event == "trade_closed" and a numeric `ret`.
    `small_sample` is True below 30 trades: treat such numbers as noise.
    """
    closed = [
        e for e in events
        if e.get("event") == "trade_closed" and isinstance(e.get("ret"), (int, float))
    ]
    wins = sum(1 for e in closed if e["ret"] > 0)
    reasons = Counter(str(e.get("reason", "unknown")) for e in closed)
    total = len(closed)
    return {
        "trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": wins / total if total else 0.0,
        "avg_return": sum(e["ret"] for e in closed) / total if total else 0.0,
        "exit_reasons": dict(reasons),
        "skips": sum(1 for e in events if e.get("event") == "skip"),
        "small_sample": total < 30,
    }
