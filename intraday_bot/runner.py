"""One paper-trading cycle for the intraday sleeve.

The runner is deliberately stateless between invocations: it reconstructs the
day from Alpaca bars, the journal, and ``state.json``. That makes it safe to run
from a scheduler every 15 minutes and safe to re-run after a failure.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .bands import (
    bands_for_slot,
    group_sessions,
    prepare_session,
    session_close,
    session_open,
    sigma_by_slot,
)
from .broker import AlpacaBroker, fetch_window
from .config import IntradaySettings
from .journal import load, record
from .strategy import Decision, Position, decide

ET = ZoneInfo("America/New_York")


def read_state(path: str | Path) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        obj = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return obj if isinstance(obj, dict) else {}


def write_state(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(target)


def run_cycle(
    s: IntradaySettings | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    broker: AlpacaBroker | None = None,
) -> dict:
    s = s or IntradaySettings.from_env()
    now_utc = now or datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    events = load(s.journal_path)
    state = read_state(s.state_path)

    if broker is None and not (s.api_key and s.api_secret):
        message = "Missing ALPACA_API_KEY / ALPACA_API_SECRET; paper bot did not run."
        record(s.journal_path, "error", message=message)
        write_state(s.state_path, {**state, "updated_at": now_et.isoformat(), "market_open": False, "error": message})
        return {"ok": False, "reason": "missing_credentials", "message": message}

    broker = broker or AlpacaBroker(s.api_key, s.api_secret, paper=s.paper)
    try:
        clock = broker.clock()
        market_open = bool(clock.is_open)
    except Exception as exc:
        message = f"Could not read the Alpaca clock: {exc}"
        record(s.journal_path, "error", message=message)
        write_state(s.state_path, {**state, "updated_at": now_et.isoformat(), "market_open": False, "error": message})
        return {"ok": False, "reason": "clock_error", "message": message}

    if not market_open:
        payload = {
            **state,
            "updated_at": now_et.isoformat(),
            "market_open": False,
            "last_action": "market_closed",
            "last_reason": "market_closed",
            "symbol": s.symbol,
        }
        write_state(s.state_path, payload)
        record(s.journal_path, "run", market_open=False, action="flat", reason="market_closed")
        return {"ok": True, "reason": "market_closed", "state": payload}

    try:
        account = broker.account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
    except Exception as exc:
        message = f"Could not read the Alpaca account: {exc}"
        record(s.journal_path, "error", message=message)
        write_state(s.state_path, {**state, "updated_at": now_et.isoformat(), "market_open": True, "error": message})
        return {"ok": False, "reason": "account_error", "message": message}

    start, end = fetch_window(now_utc, s.band_lookback_sessions + 5)
    try:
        bars = broker.bars(s.symbol, start, end, s.feed, s.bar_minutes, now=now_utc)
    except Exception as exc:
        message = f"Could not read bars: {exc}"
        record(s.journal_path, "error", message=message)
        write_state(s.state_path, {**state, "updated_at": now_et.isoformat(), "market_open": True, "error": message})
        return {"ok": False, "reason": "data_error", "message": message}

    sessions = group_sessions(bars)
    prior = [frame for day, frame in sorted(sessions.items()) if day < today]
    if today not in sessions or len(prior) < s.band_lookback_sessions:
        reason = "no_session_bars" if today not in sessions else "warmup"
        payload = {
            "updated_at": now_et.isoformat(),
            "market_open": True,
            "last_action": "flat",
            "last_reason": reason,
            "symbol": s.symbol,
            "sessions_available": len(prior),
            "sessions_required": s.band_lookback_sessions,
        }
        write_state(s.state_path, payload)
        record(s.journal_path, "skip", reason=reason, sessions_available=len(prior))
        return {"ok": True, "reason": reason, "state": payload}

    today_bars = sessions[today]
    prepared = prepare_session(today_bars, s.bar_minutes)
    if prepared.empty:
        payload = {"updated_at": now_et.isoformat(), "market_open": True, "last_action": "flat", "last_reason": "no_bars"}
        write_state(s.state_path, payload)
        record(s.journal_path, "skip", reason="no_bars")
        return {"ok": True, "reason": "no_bars", "state": payload}

    last = prepared.iloc[-1]
    last_ts = pd.Timestamp(prepared.index[-1])
    bar_end = last_ts + pd.Timedelta(minutes=s.bar_minutes)
    price = float(last["close"])
    slot = int(last["slot"])
    vwap = float(last["vwap"])
    data_age_min = max((now_et - bar_end).total_seconds() / 60.0, 0.0)

    sigma = sigma_by_slot(prior, s.bar_minutes, s.band_lookback_sessions)
    open_price = session_open(today_bars)
    prior_close = session_close(prior[-1])
    bands = bands_for_slot(open_price, prior_close, sigma, slot, s.band_mult)
    if bands is None:
        payload = {
            "updated_at": now_et.isoformat(),
            "market_open": True,
            "last_action": "flat",
            "last_reason": "bands_unavailable",
            "symbol": s.symbol,
        }
        write_state(s.state_path, payload)
        record(s.journal_path, "skip", reason="bands_unavailable")
        return {"ok": True, "reason": "bands_unavailable", "state": payload}

    raw_position = broker.position(s.symbol)
    position = _position_from(raw_position, state, bands, now_et)
    trades_today = _trades_today(events, today)
    daily_halt = _daily_loss_breached(equity, last_equity, s.max_daily_loss_pct)
    stale = data_age_min > s.max_data_age_minutes

    decision = decide(
        bar_end.to_pydatetime(),
        price,
        bands,
        vwap,
        position,
        trades_today,
        equity,
        s,
    )

    action = decision.action
    reason = decision.reason
    executed: dict | None = None

    if position is not None and (daily_halt or (decision.action == "exit_long")):
        close_reason = "daily_loss_halt" if daily_halt else decision.reason
        if dry_run:
            executed = {"action": "exit_long", "reason": close_reason, "dry_run": True}
        else:
            executed = _close_and_log(
                broker,
                s,
                position,
                now_et,
                today,
                close_reason,
                price,
                trade_id=(state.get("position") or {}).get("trade_id"),
            )
            if executed.get("action") == "exit_long":
                action, reason = "exit_long", close_reason
                state.pop("position", None)
            elif executed.get("action") == "exit_partial":
                action, reason = "exit_partial", close_reason
            else:
                action, reason = "hold", executed.get("reason", "exit_not_filled")
    elif position is None and decision.action == "enter_long":
        if daily_halt:
            action, reason = "flat", "daily_loss_halt"
        elif stale:
            action, reason = "flat", "stale_data"
        elif _entry_exists_today(broker, s.symbol, today):
            action, reason = "flat", "entry_order_exists"
        elif dry_run:
            executed = {"action": "enter_long", "qty": decision.qty, "dry_run": True}
        else:
            executed = _enter_and_log(broker, s, decision, bands, vwap, price, now_et, today)
            if executed.get("action") == "enter_long":
                state["position"] = {
                    "trade_id": executed["trade_id"],
                    "entry_price": executed["entry_price"],
                    "entry_ts": now_et.isoformat(),
                    "stop": executed["stop"],
                    "initial_stop": executed.get("initial_stop", executed["stop"]),
                    "band": bands.upper,
                    "vwap": vwap,
                }
                action, reason = "enter_long", "band_break"
            else:
                action, reason = "flat", executed.get("reason", "entry_not_filled")
    elif position is not None and decision.stop is not None:
        state.setdefault("position", {})["stop"] = decision.stop
        state["position"]["band"] = bands.upper
        state["position"]["vwap"] = vwap

    if executed and not dry_run:
        events = load(s.journal_path)
        trades_today = _trades_today(events, today)

    position_after = _position_payload(broker, s.symbol, state, price, bands, vwap)
    realized_today = _realized_today(events, today)
    payload = {
        "updated_at": now_et.isoformat(),
        "market_open": True,
        "symbol": s.symbol,
        "last_bar": bar_end.isoformat(),
        "data_age_min": round(data_age_min, 2),
        "price": price,
        "open": open_price,
        "prior_close": prior_close,
        "upper": bands.upper,
        "lower": bands.lower,
        "sigma": bands.sigma,
        "vwap": vwap,
        "slot": slot,
        "equity": equity,
        "last_equity": last_equity,
        "starting_equity": s.starting_equity,
        "today_pnl": equity - last_equity,
        "realized_today": realized_today,
        "trades_today": trades_today,
        "daily_loss_halt": daily_halt,
        "stale_data": stale,
        "last_action": action,
        "last_reason": reason,
        "position": position_after,
        "executed": executed,
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
    write_state(s.state_path, payload)
    record(
        s.journal_path,
        "run",
        market_open=True,
        symbol=s.symbol,
        price=price,
        upper=bands.upper,
        lower=bands.lower,
        vwap=vwap,
        sigma=bands.sigma,
        slot=slot,
        equity=equity,
        last_equity=last_equity,
        trades_today=trades_today,
        position_qty=position_after["qty"] if position_after else 0,
        action=action,
        reason=reason,
        daily_loss_halt=daily_halt,
        stale_data=stale,
        data_age_min=round(data_age_min, 2),
    )
    return {"ok": True, "action": action, "reason": reason, "state": payload, "executed": executed}


def _position_from(raw, state: dict, bands, now_et: datetime) -> Position | None:
    if raw is None:
        return None
    try:
        qty = int(float(raw.qty))
    except (TypeError, ValueError, AttributeError):
        return None
    if qty <= 0:
        return None
    saved = state.get("position") or {}
    entry = float(saved.get("entry_price") or getattr(raw, "avg_entry_price", 0.0) or 0.0)
    if entry <= 0:
        entry = float(getattr(raw, "current_price", 0.0) or 0.0)
    stop = float(saved.get("stop") or max(bands.upper, 0.0))
    initial_stop = float(saved.get("initial_stop") or stop)
    entry_ts = saved.get("entry_ts") or now_et.isoformat()
    try:
        entry_time = datetime.fromisoformat(str(entry_ts))
    except ValueError:
        entry_time = now_et
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=ET)

    return Position(
        qty=qty,
        entry_price=entry,
        entry_time=entry_time,
        stop=stop,
        band=float(saved.get("band") or bands.upper),
        vwap=float(saved.get("vwap") or 0.0),
        side="long",
        initial_stop=initial_stop,
    )


def _position_payload(broker: AlpacaBroker, symbol: str, state: dict, price: float, bands, vwap: float) -> dict | None:
    raw = broker.position(symbol)
    if raw is None:
        return None
    try:
        qty = int(float(raw.qty))
    except (TypeError, ValueError, AttributeError):
        return None
    if qty <= 0:
        return None
    saved = state.get("position") or {}
    entry = float(saved.get("entry_price") or getattr(raw, "avg_entry_price", 0.0) or price)
    stop = float(saved.get("stop") or max(bands.upper, vwap))
    initial_stop = float(saved.get("initial_stop") or stop)
    unrealized = float(getattr(raw, "unrealized_pl", 0.0) or 0.0)
    return {
        "symbol": symbol,
        "side": "long",
        "qty": qty,
        "entry_price": entry,
        "current_price": price,
        "stop": stop,
        "initial_stop": initial_stop,
        "band": bands.upper,
        "vwap": vwap,
        "unrealized_pl": unrealized,
        "unrealized_pct": (price / entry - 1.0) if entry > 0 else 0.0,
        "entry_ts": saved.get("entry_ts"),
        "trade_id": saved.get("trade_id"),
    }


def _enter_and_log(
    broker: AlpacaBroker,
    s: IntradaySettings,
    decision: Decision,
    bands,
    vwap: float,
    price: float,
    now_et: datetime,
    today,
) -> dict:
    assert decision.qty is not None and decision.stop is not None
    client_order_id = f"intraday-{today}-{now_et.strftime('%H%M')}-{uuid.uuid4().hex[:8]}"
    order = broker.submit_market(s.symbol, decision.qty, "buy", client_order_id)
    filled = broker.wait_for_fill(order.id)
    status = _enum_value(getattr(filled, "status", ""))
    filled_qty = int(float(getattr(filled, "filled_qty", 0) or 0))
    if status not in {"filled", "partially_filled"} or filled_qty <= 0:
        try:
            broker.cancel_open_orders(s.symbol)
        except Exception:
            pass
        message = f"Entry order {order.id} not filled (status={status or 'unknown'})."
        record(s.journal_path, "error", message=message, order_id=str(order.id))
        return {"action": "entry_not_filled", "reason": "entry_not_filled", "order_id": str(order.id), "status": status}
    qty = min(filled_qty, decision.qty)
    entry_price = _fill_price(filled, price)
    record(
        s.journal_path,
        "entry",
        trade_id=client_order_id,
        order_id=str(order.id),
        symbol=s.symbol,
        side="long",
        qty=qty,
        price=entry_price,
        signal_price=price,
        stop=decision.stop,
        initial_stop=decision.stop,
        upper=bands.upper,
        lower=bands.lower,
        vwap=vwap,
        risk_amount=decision.risk_amount,
        date=str(today),
        entry_ts=now_et.isoformat(),
    )
    return {
        "action": "enter_long",
        "trade_id": client_order_id,
        "qty": qty,
        "entry_price": entry_price,
        "stop": decision.stop,
        "initial_stop": decision.stop,
        "order_id": str(order.id),
    }


def _close_and_log(
    broker: AlpacaBroker,
    s: IntradaySettings,
    position: Position,
    now_et: datetime,
    today,
    reason: str,
    fallback_price: float,
    trade_id: str | None = None,
) -> dict:
    order = broker.close_position(s.symbol)
    if order is None:
        message = "Broker returned no close order; position is still open."
        record(s.journal_path, "error", message=message)
        return {"action": "exit_not_filled", "reason": "exit_not_filled", "message": message}
    filled = broker.wait_for_fill(order.id)
    status = _enum_value(getattr(filled, "status", ""))
    filled_qty = int(float(getattr(filled, "filled_qty", 0) or 0))
    if status not in {"filled", "partially_filled"} or filled_qty <= 0:
        message = f"Close order {order.id} not filled (status={status or 'unknown'})."
        record(s.journal_path, "error", message=message, order_id=str(order.id))
        return {"action": "exit_not_filled", "reason": "exit_not_filled", "order_id": str(order.id), "status": status}
    exit_price = _fill_price(filled, fallback_price)
    qty = min(filled_qty, position.qty)
    initial_stop = position.initial_stop or position.stop
    pnl = (exit_price - position.entry_price) * qty - s.commission_per_share * 2 * qty
    risk_per_share = position.entry_price - initial_stop
    r_multiple = (exit_price - position.entry_price) / risk_per_share if risk_per_share > 0 else 0.0
    hold_minutes = max((now_et - position.entry_time).total_seconds() / 60.0, 0.0)
    if trade_id is None:
        for event in reversed(load(s.journal_path)):
            if event.get("event") == "entry" and int(event.get("qty", 0)) == qty:
                trade_id = event.get("trade_id")
                break
    record(
        s.journal_path,
        "trade_closed",
        trade_id=trade_id,
        symbol=s.symbol,
        side="long",
        qty=qty,
        entry_price=position.entry_price,
        exit_price=exit_price,
        stop=initial_stop,
        pnl=pnl,
        ret=(exit_price / position.entry_price - 1.0) if position.entry_price > 0 else 0.0,
        r_multiple=r_multiple,
        reason=reason,
        hold_minutes=hold_minutes,
        date=str(today),
        entry_ts=position.entry_time.isoformat(),
        exit_ts=now_et.isoformat(),
    )
    remaining = max(position.qty - qty, 0)
    return {
        "action": "exit_long" if remaining == 0 else "exit_partial",
        "reason": reason,
        "qty": qty,
        "remaining_qty": remaining,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "r_multiple": r_multiple,
    }


def _fill_price(order, fallback: float) -> float:
    if order is None:
        return fallback
    value = getattr(order, "filled_avg_price", None)
    try:
        price = float(value)
    except (TypeError, ValueError):
        return fallback
    return price if price > 0 else fallback


def _enum_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


def _entry_exists_today(broker: AlpacaBroker, symbol: str, today) -> bool:
    session_start = datetime.combine(today, time(9, 30), tzinfo=ET)
    try:
        orders = broker.orders_since(symbol, session_start)
    except Exception:
        return False
    for order in orders:
        client_id = str(getattr(order, "client_order_id", "") or "")
        if not client_id.startswith("intraday-"):
            continue
        if _enum_value(getattr(order, "side", "")) != "buy":
            continue
        if _enum_value(getattr(order, "status", "")) in {"canceled", "rejected", "expired"}:
            continue
        return True
    return False


def _trades_today(events: list[dict], today) -> int:
    return sum(
        1
        for e in events
        if e.get("event") == "entry" and str(e.get("date")) == str(today)
    )


def _realized_today(events: list[dict], today) -> float:
    return sum(
        float(e.get("pnl", 0.0))
        for e in events
        if e.get("event") == "trade_closed" and str(e.get("date")) == str(today)
    )


def _daily_loss_breached(equity: float, last_equity: float, max_loss_pct: float) -> bool:
    if last_equity <= 0:
        return False
    return (equity - last_equity) / last_equity <= -max_loss_pct
