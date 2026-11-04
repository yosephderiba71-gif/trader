from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from intraday_bot.bands import Bands
from intraday_bot.config import IntradaySettings
from intraday_bot.journal import load
from intraday_bot.runner import _close_and_log, _enter_and_log, _position_from
from intraday_bot.strategy import Decision, Position

ET = ZoneInfo("America/New_York")


class FakeOrder:
    def __init__(self, order_id="o-1", status="filled", filled_qty=10, filled_avg_price=101.0):
        self.id = order_id
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class FakeBroker:
    def __init__(self, order):
        self.order = order
        self.canceled = False

    def submit_market(self, symbol, qty, side, client_order_id):
        return self.order

    def wait_for_fill(self, order_id, timeout=15.0):
        return self.order

    def cancel_open_orders(self, symbol):
        self.canceled = True

    def close_position(self, symbol):
        return self.order


def test_position_from_uses_saved_initial_stop():
    raw = type("Raw", (), {"qty": "10", "avg_entry_price": "101", "current_price": "102"})()
    bands = Bands(upper=100.5, lower=99.0, sigma=0.01, anchor_high=100.0, anchor_low=99.0)
    state = {
        "position": {
            "entry_price": 101.0,
            "stop": 100.5,
            "initial_stop": 99.0,
            "entry_ts": "2026-11-02T11:00:00",
            "band": 100.5,
            "vwap": 100.2,
        }
    }
    pos = _position_from(raw, state, bands, datetime(2026, 11, 2, 12, 0, tzinfo=ET))
    assert pos is not None
    assert pos.initial_stop == pytest.approx(99.0)
    assert pos.stop == pytest.approx(100.5)
    assert pos.entry_time.tzinfo is not None


def test_close_uses_initial_stop_for_r_multiple(tmp_path):
    journal = tmp_path / "journal.jsonl"
    s = IntradaySettings(journal_path=str(journal), commission_per_share=0.0)
    broker = FakeBroker(FakeOrder(status="filled", filled_qty=10, filled_avg_price=102.0))
    position = Position(
        qty=10,
        entry_price=100.0,
        entry_time=datetime(2026, 11, 2, 11, 0, tzinfo=ET),
        stop=101.0,
        band=100.5,
        vwap=100.2,
        initial_stop=98.0,
    )
    result = _close_and_log(
        broker,
        s,
        position,
        datetime(2026, 11, 2, 12, 0, tzinfo=ET),
        date(2026, 11, 2),
        "trail_stop",
        102.0,
        trade_id="t-1",
    )
    assert result["action"] == "exit_long"
    assert result["r_multiple"] == pytest.approx(1.0)
    closed = [e for e in load(journal) if e.get("event") == "trade_closed"]
    assert len(closed) == 1
    assert closed[0]["stop"] == pytest.approx(98.0)
    assert closed[0]["trade_id"] == "t-1"


def test_close_rejects_unfilled_order(tmp_path):
    journal = tmp_path / "journal.jsonl"
    s = IntradaySettings(journal_path=str(journal))
    broker = FakeBroker(FakeOrder(status="accepted", filled_qty=0))
    position = Position(
        qty=10,
        entry_price=100.0,
        entry_time=datetime(2026, 11, 2, 11, 0, tzinfo=ET),
        stop=99.0,
        band=100.5,
        vwap=100.2,
        initial_stop=98.0,
    )
    result = _close_and_log(
        broker,
        s,
        position,
        datetime(2026, 11, 2, 12, 0, tzinfo=ET),
        date(2026, 11, 2),
        "trail_stop",
        102.0,
    )
    assert result["action"] == "exit_not_filled"
    assert not [e for e in load(journal) if e.get("event") == "trade_closed"]


def test_enter_rejects_unfilled_order(tmp_path):
    journal = tmp_path / "journal.jsonl"
    s = IntradaySettings(journal_path=str(journal))
    broker = FakeBroker(FakeOrder(status="accepted", filled_qty=0))
    decision = Decision(
        action="enter_long",
        reason="band_break",
        stop=99.0,
        qty=10,
        risk_amount=20.0,
        upper=100.5,
        lower=98.0,
    )
    bands = Bands(upper=100.5, lower=98.0, sigma=0.01, anchor_high=100.0, anchor_low=99.0)
    result = _enter_and_log(
        broker,
        s,
        decision,
        bands,
        100.2,
        101.0,
        datetime(2026, 11, 2, 10, 0, tzinfo=ET),
        date(2026, 11, 2),
    )
    assert result["action"] == "entry_not_filled"
    assert broker.canceled is True
    assert not [e for e in load(journal) if e.get("event") == "entry"]
    assert [e for e in load(journal) if e.get("event") == "error"]


class CycleOrder(FakeOrder):
    def __init__(self):
        super().__init__(order_id="cycle-1", status="filled", filled_qty=0, filled_avg_price=101.0)


class RawPosition:
    def __init__(self, qty, entry, current):
        self.qty = str(qty)
        self.avg_entry_price = str(entry)
        self.current_price = str(current)
        self.unrealized_pl = str((current - entry) * qty)


class CycleBroker:
    def __init__(self, bars):
        self.bars_frame = bars
        self.order = CycleOrder()
        self.position_obj = None
        self.submitted = []

    def clock(self):
        return type("Clock", (), {"is_open": True})()

    def account(self):
        return type("Account", (), {"equity": "100000", "last_equity": "100000"})()

    def position(self, symbol):
        return self.position_obj

    def bars(self, symbol, start, end, feed, bar_minutes, now=None):
        return self.bars_frame

    def submit_market(self, symbol, qty, side, client_order_id):
        self.order.filled_qty = qty
        self.submitted.append((symbol, qty, side, client_order_id))
        self.position_obj = RawPosition(qty, self.order.filled_avg_price, 101.5)
        return self.order

    def wait_for_fill(self, order_id, timeout=15.0):
        return self.order

    def cancel_open_orders(self, symbol):
        return None

    def orders_since(self, symbol, after):
        return []


def test_run_cycle_dry_run_and_live_entry(tmp_path):
    import pandas as pd

    from intraday_bot.demo import make_bars
    from intraday_bot.runner import read_state, run_cycle

    journal = tmp_path / "journal.jsonl"
    state_path = tmp_path / "state.json"
    s = IntradaySettings(journal_path=str(journal), state_path=str(state_path))
    bars = make_bars(sessions=20, seed=11)
    today = bars.index[-1].date()
    now = datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=10, minute=0)
    completed = bars[bars.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]
    mask = (
        (completed.index.date == today)
        & (completed.index.hour == 9)
        & (completed.index.minute == 55)
    )
    completed = completed.copy()
    completed.loc[mask, "close"] = completed.loc[mask, "open"] * 1.02
    completed.loc[mask, "high"] = completed.loc[mask, ["high", "close"]].max(axis=1)

    dry_broker = CycleBroker(completed)
    dry = run_cycle(s, now=now, dry_run=True, broker=dry_broker)
    assert dry["ok"] is True
    assert dry["action"] == "enter_long"
    assert dry["executed"]["dry_run"] is True
    assert dry_broker.submitted == []
    assert not [e for e in load(journal) if e.get("event") == "entry"]

    live_broker = CycleBroker(completed)
    live = run_cycle(s, now=now, dry_run=False, broker=live_broker)
    assert live["ok"] is True
    assert live["action"] == "enter_long"
    assert len(live_broker.submitted) == 1
    assert live["state"]["trades_today"] == 1
    assert live["state"]["position"]["trade_id"]
    events = load(journal)
    assert len([e for e in events if e.get("event") == "entry"]) == 1
    assert read_state(state_path)["position"]["trade_id"]


def test_close_partial_fill_keeps_remaining_position(tmp_path):
    journal = tmp_path / "journal.jsonl"
    s = IntradaySettings(journal_path=str(journal), commission_per_share=0.0)
    broker = FakeBroker(FakeOrder(status="partially_filled", filled_qty=4, filled_avg_price=102.0))
    position = Position(
        qty=10,
        entry_price=100.0,
        entry_time=datetime(2026, 11, 2, 11, 0, tzinfo=ET),
        stop=101.0,
        band=100.5,
        vwap=100.2,
        initial_stop=98.0,
    )
    result = _close_and_log(
        broker,
        s,
        position,
        datetime(2026, 11, 2, 12, 0, tzinfo=ET),
        date(2026, 11, 2),
        "trail_stop",
        102.0,
        trade_id="t-1",
    )
    assert result["action"] == "exit_partial"
    assert result["remaining_qty"] == 6
    closed = [e for e in load(journal) if e.get("event") == "trade_closed"]
    assert len(closed) == 1
    assert closed[0]["qty"] == 4
