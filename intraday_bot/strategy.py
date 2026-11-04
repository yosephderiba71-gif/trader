"""Pure decision logic for the long-only intraday sleeve.

Rules (paper variant "current band + VWAP", long side only):

- Enter long at a half-hourly check when price is above the upper noise band.
- Initial and trailing stop: ``max(upper_band, VWAP)``.
- Exit on the stop, at the flat time, or at the end of the session.
- At most one trade per day in v1.

No broker or data imports live here, so the rules are testable offline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .bands import Bands, is_decision_slot, slot_time
from .config import IntradaySettings

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Position:
    qty: int
    entry_price: float
    entry_time: datetime
    stop: float
    band: float
    vwap: float
    side: str = "long"
    initial_stop: float | None = None


@dataclass(frozen=True)
class Decision:
    action: str  # enter_long | exit_long | hold | flat
    reason: str
    stop: float | None = None
    qty: int | None = None
    risk_amount: float | None = None
    upper: float | None = None
    lower: float | None = None


def size_position(
    equity: float,
    price: float,
    stop: float,
    s: IntradaySettings,
) -> tuple[int, float] | None:
    """Whole-share size so a stop-out costs about ``risk_per_trade`` of equity.

    The notional cap also binds, so a very tight stop cannot create absurd
    leverage. Returns ``None`` when even one share would exceed the risk budget.
    """
    if equity <= 0 or price <= 0 or stop <= 0 or stop >= price:
        return None
    risk_per_share = price - stop
    risk_budget = equity * s.risk_per_trade
    by_risk = math.floor(risk_budget / risk_per_share)
    by_cap = math.floor(equity * s.max_position_pct / price)
    qty = min(by_risk, by_cap)
    if qty < 1:
        return None
    return qty, qty * risk_per_share


def decide(
    now: datetime,
    price: float,
    bands: Bands,
    vwap: float,
    position: Position | None,
    trades_today: int,
    equity: float,
    s: IntradaySettings,
) -> Decision:
    """Evaluate one half-hourly decision point."""
    now_et = now.astimezone(ET) if now.tzinfo else now.replace(tzinfo=ET)
    slot = now_et.hour * 60 + now_et.minute

    if position is not None and position.qty > 0:
        if position.side == "long":
            if slot >= _minutes(s.flat_time):
                return Decision("exit_long", "flat_by_close", stop=position.stop, upper=bands.upper, lower=bands.lower)
            if not is_decision_slot(slot, s.entry_start, s.entry_end, s.decision_minutes):
                return Decision("hold", "between_checks", stop=position.stop, upper=bands.upper, lower=bands.lower)
            stop = max(position.stop, bands.upper, vwap)
            if price < stop:
                return Decision("exit_long", "trail_stop", stop=stop, upper=bands.upper, lower=bands.lower)
            return Decision("hold", "trailing", stop=stop, upper=bands.upper, lower=bands.lower)

    if slot >= _minutes(s.flat_time):
        return Decision("flat", "after_flat", upper=bands.upper, lower=bands.lower)
    if slot < _minutes(s.entry_start):
        return Decision("flat", "before_window", upper=bands.upper, lower=bands.lower)
    if slot > _minutes(s.entry_end):
        return Decision("flat", "after_window", upper=bands.upper, lower=bands.lower)
    if not is_decision_slot(slot, s.entry_start, s.entry_end, s.decision_minutes):
        return Decision("flat", "not_decision_time", upper=bands.upper, lower=bands.lower)
    if trades_today >= s.max_trades_per_day:
        return Decision("flat", "max_trades", upper=bands.upper, lower=bands.lower)

    if price > bands.upper:
        stop = max(bands.upper, vwap)
        if stop >= price:
            return Decision("flat", "stop_above_price", upper=bands.upper, lower=bands.lower)
        sized = size_position(equity, price, stop, s)
        if sized is None:
            return Decision("flat", "size_zero", upper=bands.upper, lower=bands.lower)
        qty, risk_amount = sized
        return Decision(
            "enter_long",
            "band_break",
            stop=stop,
            qty=qty,
            risk_amount=risk_amount,
            upper=bands.upper,
            lower=bands.lower,
        )

    if price < bands.lower and s.allow_short:
        # Short side is intentionally not implemented in v1.
        return Decision("flat", "short_not_implemented", upper=bands.upper, lower=bands.lower)
    if price < bands.lower:
        return Decision("flat", "short_signal_ignored", upper=bands.upper, lower=bands.lower)
    return Decision("flat", "inside_band", upper=bands.upper, lower=bands.lower)


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def decision_slot_label(slot: int) -> str:
    return slot_time(slot).strftime("%H:%M")
