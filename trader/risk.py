"""Position sizing and portfolio-level guards. Pure functions, no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class EntryPlan:
    qty: int
    stop_price: float
    target_price: float
    risk_amount: float


def plan_entry(equity: float, price: float, atr_value: float, s: Settings) -> EntryPlan | None:
    """Size a long entry so that hitting the stop loses about risk_per_trade of equity.

    Whole shares only: bracket orders do not support fractional quantities.
    """
    if equity <= 0 or price <= 0 or not math.isfinite(atr_value) or atr_value <= 0:
        return None
    stop_distance = s.atr_stop_mult * atr_value
    stop = round(price - stop_distance, 2)
    if stop <= 0:
        return None
    target = round(price + s.atr_target_mult * atr_value, 2)

    risk_amount = equity * s.risk_per_trade
    by_risk = math.floor(risk_amount / (price - stop))
    by_cap = math.floor(equity * s.max_position_pct / price)
    qty = min(by_risk, by_cap)
    if qty < 1:
        return None
    return EntryPlan(qty=qty, stop_price=stop, target_price=target, risk_amount=risk_amount)


def daily_loss_breached(equity: float, last_equity: float, max_loss_pct: float) -> bool:
    """True when equity is down more than max_loss_pct versus the previous session close."""
    if last_equity <= 0:
        return False
    return (equity - last_equity) / last_equity <= -max_loss_pct
