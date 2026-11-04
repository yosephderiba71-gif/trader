"""Generate a deterministic synthetic journal so the dashboard can be previewed.

This is not a performance claim. It exists so the front end has realistic-shaped
data before the paper bot has traded for a few weeks.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import IntradaySettings
from .dashboard import build_payload, write_static


def make_bars(sessions: int = 260, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="America/New_York").normalize()
    days = pd.bdate_range(end=end, periods=sessions, tz="America/New_York")
    frames = []
    prev_close = 470.0
    for day in days:
        start = day + pd.Timedelta(hours=9, minutes=30)
        end = day + pd.Timedelta(hours=15, minutes=55)
        index = pd.date_range(start, end, freq="5min")
        n = len(index)
        trend = rng.choice([-1.0, 0.0, 1.0], p=[0.18, 0.64, 0.18])
        drift = trend * rng.uniform(0.00002, 0.00012)
        vol = rng.uniform(0.0004, 0.0012)
        returns = rng.normal(drift, vol, n)
        returns[1:] -= 0.10 * returns[:-1]
        gap = rng.normal(0.0, 0.0006)
        close = prev_close * np.exp(np.cumsum(returns) + gap)
        open_ = np.r_[prev_close * np.exp(gap), close[:-1]]
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.0006, n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.0006, n)))
        volume = rng.integers(250_000, 2_500_000, n).astype(float)
        frames.append(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=index,
            )
        )
        prev_close = float(close[-1])
    return pd.concat(frames).sort_index()


def write_demo(
    journal_path: str | Path = "intraday/demo_journal.jsonl",
    state_path: str | Path = "intraday/demo_state.json",
    html_path: str | Path = "intraday_demo.html",
    sessions: int = 260,
    seed: int = 11,
) -> dict:
    s = IntradaySettings()
    bars = make_bars(sessions=sessions, seed=seed)
    result = run_backtest(bars, s, starting_equity=s.starting_equity)
    trades = result.trades

    journal = Path(journal_path)
    journal.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for _, t in trades.iterrows():
        entry = {
            "ts": t["entry_ts"],
            "event": "entry",
            "trade_id": t["trade_id"],
            "symbol": s.symbol,
            "side": "long",
            "qty": int(t["qty"]),
            "price": float(t["entry_price"]),
            "signal_price": float(t["entry_price"]),
            "stop": float(t["stop"]),
            "date": str(t["date"]),
            "entry_ts": t["entry_ts"],
        }
        closed = {
            "ts": t["exit_ts"],
            "event": "trade_closed",
            "trade_id": t["trade_id"],
            "symbol": s.symbol,
            "side": "long",
            "qty": int(t["qty"]),
            "entry_price": float(t["entry_price"]),
            "exit_price": float(t["exit_price"]),
            "stop": float(t["stop"]),
            "pnl": float(t["pnl"]),
            "ret": float(t["ret"]),
            "r_multiple": float(t["r_multiple"]),
            "reason": str(t["reason"]),
            "hold_minutes": float(t["hold_minutes"]),
            "mae": float(t["mae"]),
            "mfe": float(t["mfe"]),
            "date": str(t["date"]),
            "entry_ts": t["entry_ts"],
            "exit_ts": t["exit_ts"],
        }
        lines.append(json.dumps(entry, default=str))
        lines.append(json.dumps(closed, default=str))
    journal.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    net = float(trades["pnl"].sum()) if not trades.empty else 0.0
    last_close = float(bars["close"].iloc[-1])
    last_day = bars.index[-1].date()
    today_trades = trades[trades["date"].astype(str) == str(last_day)] if not trades.empty else trades
    realized_today = float(today_trades["pnl"].sum()) if not today_trades.empty else 0.0
    open_entry = last_close - 0.35
    open_qty = 120
    open_unrealized = (last_close - open_entry) * open_qty
    equity = s.starting_equity + net + open_unrealized
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    last_bar = now_et.replace(second=0, microsecond=0) - pd.Timedelta(minutes=5)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "market_open": True,
        "symbol": s.symbol,
        "last_bar": last_bar.isoformat(),
        "data_age_min": 1.0,
        "price": last_close,
        "open": float(bars["open"].iloc[-1]),
        "prior_close": float(bars["close"].iloc[-2]),
        "upper": last_close + 0.15,
        "lower": last_close - 2.10,
        "vwap": last_close - 0.25,
        "sigma": 0.0038,
        "equity": equity,
        "last_equity": equity - realized_today - open_unrealized,
        "starting_equity": s.starting_equity,
        "today_pnl": realized_today + open_unrealized,
        "realized_today": realized_today,
        "trades_today": 1,
        "daily_loss_halt": False,
        "stale_data": False,
        "last_action": "hold",
        "last_reason": "trailing",
        "position": {
            "symbol": s.symbol,
            "side": "long",
            "qty": open_qty,
            "entry_price": open_entry,
            "current_price": last_close,
            "stop": open_entry - 0.60,
            "initial_stop": open_entry - 0.60,
            "band": last_close + 0.15,
            "vwap": last_close - 0.25,
            "unrealized_pl": open_unrealized,
            "unrealized_pct": last_close / open_entry - 1.0,
            "entry_ts": (last_bar - pd.Timedelta(hours=2)).isoformat(),
            "trade_id": "demo-open",
        },
        "strategy": {
            "name": "SPY intraday momentum (long-only)",
            "band_lookback_sessions": s.band_lookback_sessions,
            "band_mult": s.band_mult,
            "risk_per_trade": s.risk_per_trade,
            "max_position_pct": s.max_position_pct,
            "max_trades_per_day": s.max_trades_per_day,
            "max_daily_loss_pct": s.max_daily_loss_pct,
            "flat_time": s.flat_time.strftime("%H:%M"),
        },
    }
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state_path).write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")

    payload = build_payload(journal_path, state_path, starting_equity=s.starting_equity, demo=True)
    write_static(payload, html_path)
    return {
        "trades": int(len(trades)),
        "net_pnl": net,
        "sessions": int(result.sessions),
        "journal": str(journal),
        "state": str(state_path),
        "html": str(html_path),
        "metrics": result.metrics,
    }


if __name__ == "__main__":
    summary = write_demo()
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2, default=str))
