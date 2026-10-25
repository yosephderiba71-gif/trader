"""One trading cycle: manage exits, then consider new entries.

The agent is stateless. The broker (Alpaca) is the source of truth for positions and
orders, so a restarted or overlapping run cannot double-buy. Dependencies are injected,
which lets the tests run without any network access.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import pandas as pd

from .config import Settings
from .research import evaluate_all, qualifying_best
from .risk import EntryPlan, daily_loss_breached, plan_entry
from .strategies import STRATEGIES, atr


def log(event: str, **fields) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **fields}
    print(json.dumps(record, default=str), flush=True)


class Broker(Protocol):
    def is_market_open(self) -> bool: ...
    def account(self): ...
    def positions(self) -> dict[str, float]: ...
    def has_order_since(self, symbol: str, since: datetime) -> bool: ...
    def submit_bracket_buy(self, symbol: str, plan: EntryPlan, client_order_id: str): ...
    def close_long(self, symbol: str): ...


@dataclass
class CycleReport:
    skipped: str | None = None
    halted: bool = False
    actions: list[dict] = field(default_factory=list)


def _last(series: pd.Series) -> bool:
    return bool(series.iloc[-1])


def run_cycle(s: Settings, broker: Broker, bars: dict[str, pd.DataFrame], execute: bool) -> CycleReport:
    report = CycleReport()
    if not broker.is_market_open():
        report.skipped = "market_closed"
        log("cycle_skipped", reason="market_closed")
        return report

    account = broker.account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    buying_power = float(account.buying_power)
    report.halted = daily_loss_breached(equity, last_equity, s.max_daily_loss_pct)
    if report.halted:
        log("halt", reason="daily_loss_limit", equity=equity, last_equity=last_equity)

    held = broker.positions()
    best = qualifying_best(evaluate_all(bars, s), s)
    log(
        "selection",
        mode="live" if execute else "dry_run",
        chosen={sym: {"strategy": c.strategy, "oos_sharpe": round(c.oos.sharpe, 2), "oos_trades": c.oos.trades} for sym, c in best.items()},
        held=sorted(held),
    )

    # 1) Exits. Allowed even when halted, because closing risk always helps.
    closed: set[str] = set()
    for sym in held:
        cand = best.get(sym)
        if cand is None:
            # The bracket stop and target still protect the position.
            log("hold", symbol=sym, reason="no_qualifying_strategy")
            continue
        df = bars.get(sym)
        if df is None or df.empty:
            continue
        if _last(STRATEGIES[cand.strategy](df).exits):
            if execute:
                broker.close_long(sym)
            closed.add(sym)
            report.actions.append({"action": "close", "symbol": sym})
            log("close", symbol=sym, strategy=cand.strategy, executed=execute)

    if report.halted:
        return report

    # 2) Entries, best out-of-sample Sharpe first.
    open_slots = s.max_open_positions - (len(held) - len(closed))
    new_entries = 0
    for cand in sorted(best.values(), key=lambda c: c.oos.sharpe, reverse=True):
        if new_entries >= s.max_new_entries_per_cycle or open_slots <= 0:
            break
        sym = cand.symbol
        if sym in held:
            continue
        df = bars.get(sym)
        if df is None or df.empty or not _last(STRATEGIES[cand.strategy](df).entries):
            continue

        bar_time = df.index[-1]
        if broker.has_order_since(sym, bar_time.to_pydatetime()):
            log("skip", symbol=sym, reason="already_traded_this_bar", bar=str(bar_time))
            continue

        price = float(df["close"].iloc[-1])
        plan = plan_entry(equity, price, float(atr(df).iloc[-1]), s)
        if plan is None:
            log("skip", symbol=sym, reason="position_too_small")
            continue
        cost = plan.qty * price
        if cost > buying_power:
            log("skip", symbol=sym, reason="insufficient_buying_power", cost=round(cost, 2))
            continue

        client_id = f"{sym}-{cand.strategy}-{bar_time:%Y%m%d%H}"
        if execute:
            broker.submit_bracket_buy(sym, plan, client_id)
        report.actions.append({"action": "buy", "symbol": sym, "qty": plan.qty})
        log(
            "buy",
            symbol=sym,
            strategy=cand.strategy,
            qty=plan.qty,
            price=price,
            stop=plan.stop_price,
            target=plan.target_price,
            risk_usd=round(plan.risk_amount, 2),
            executed=execute,
        )
        open_slots -= 1
        new_entries += 1
        buying_power -= cost

    return report
