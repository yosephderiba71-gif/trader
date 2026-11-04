"""Settings for the isolated intraday sleeve, read from ``INTRADAY_*`` env vars.

Credentials deliberately use the same names as the existing bot so a paper
account can be reused, but this module never prints them and never writes them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import time


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _time(name: str, default: str) -> time:
    raw = os.getenv(name, default)
    hour, minute = raw.split(":", 1)
    return time(int(hour), int(minute))


@dataclass(frozen=True)
class IntradaySettings:
    symbol: str = "SPY"
    feed: str = "iex"
    bar_minutes: int = 5
    band_lookback_sessions: int = 14
    band_mult: float = 1.0

    entry_start: time = time(10, 0)
    entry_end: time = time(15, 30)
    flat_time: time = time(15, 45)
    decision_minutes: int = 30

    risk_per_trade: float = 0.0025
    max_position_pct: float = 1.0
    max_trades_per_day: int = 1
    max_daily_loss_pct: float = 0.01
    max_data_age_minutes: int = 20
    allow_short: bool = False

    slippage_bps: float = 1.0
    commission_per_share: float = 0.0
    starting_equity: float = 100_000.0

    journal_path: str = "intraday/journal.jsonl"
    state_path: str = "intraday/state.json"
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True

    @classmethod
    def from_env(cls) -> "IntradaySettings":
        return cls(
            symbol=os.getenv("INTRADAY_SYMBOL", "SPY").upper(),
            feed=os.getenv("INTRADAY_FEED", "iex"),
            bar_minutes=_int("INTRADAY_BAR_MINUTES", 5),
            band_lookback_sessions=_int("INTRADAY_BAND_LOOKBACK", 14),
            band_mult=_float("INTRADAY_BAND_MULT", 1.0),
            entry_start=_time("INTRADAY_ENTRY_START", "10:00"),
            entry_end=_time("INTRADAY_ENTRY_END", "15:30"),
            flat_time=_time("INTRADAY_FLAT_TIME", "15:45"),
            decision_minutes=_int("INTRADAY_DECISION_MINUTES", 30),
            risk_per_trade=_float("INTRADAY_RISK_PER_TRADE", 0.0025),
            max_position_pct=_float("INTRADAY_MAX_POSITION_PCT", 1.0),
            max_trades_per_day=_int("INTRADAY_MAX_TRADES_PER_DAY", 1),
            max_daily_loss_pct=_float("INTRADAY_MAX_DAILY_LOSS_PCT", 0.01),
            max_data_age_minutes=_int("INTRADAY_MAX_DATA_AGE_MIN", 20),
            allow_short=os.getenv("INTRADAY_ALLOW_SHORT", "0") in {"1", "true", "yes"},
            slippage_bps=_float("INTRADAY_SLIPPAGE_BPS", 1.0),
            commission_per_share=_float("INTRADAY_COMMISSION_PER_SHARE", 0.0),
            starting_equity=_float("INTRADAY_STARTING_EQUITY", 100_000.0),
            journal_path=os.getenv("INTRADAY_JOURNAL", "intraday/journal.jsonl"),
            state_path=os.getenv("INTRADAY_STATE", "intraday/state.json"),
            api_key=os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY", ""),
            api_secret=os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET", ""),
            paper=True,
        )

    def require_credentials(self) -> None:
        if not (self.api_key and self.api_secret):
            raise SystemExit(
                "Missing ALPACA_API_KEY / ALPACA_API_SECRET. "
                "Use paper-trading keys; this bot is paper-only."
            )


def decision_times(s: IntradaySettings) -> list[time]:
    """Half-hourly decision points from ``entry_start`` through ``entry_end``."""
    out: list[time] = []
    t = s.entry_start
    while t <= s.entry_end:
        out.append(t)
        minutes = t.hour * 60 + t.minute + s.decision_minutes
        t = time(minutes // 60, minutes % 60)
    return out
