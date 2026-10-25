import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.helpers import flat_bars
from trader import agent
from trader.backtest import Metrics
from trader.config import Settings
from trader.research import Candidate
from trader.strategies import Signals

S = Settings(max_open_positions=3, max_new_entries_per_cycle=1, max_daily_loss_pct=0.02)


class FakeBroker:
    def __init__(self, market_open=True, equity=100_000.0, last_equity=100_000.0, positions=None, recent_order=False):
        self.market_open = market_open
        self.equity = equity
        self.last_equity = last_equity
        self._positions = dict(positions or {})
        self.recent_order = recent_order
        self.bought = []
        self.closed = []

    def is_market_open(self):
        return self.market_open

    def account(self):
        return SimpleNamespace(equity=str(self.equity), last_equity=str(self.last_equity), buying_power=str(self.equity))

    def positions(self):
        return dict(self._positions)

    def has_order_since(self, symbol, since):
        return self.recent_order

    def submit_bracket_buy(self, symbol, plan, client_order_id):
        self.bought.append((symbol, plan.qty, client_order_id))

    def close_long(self, symbol):
        self.closed.append(symbol)


def _fire_on_last_bar(df, entry=True, exit_=False):
    n = len(df)
    flags = pd.Series([False] * n, index=df.index)
    last = flags.copy()
    last.iloc[-1] = True
    return Signals(entries=last if entry else flags, exits=last if exit_ else flags)


@pytest.fixture
def stub_strategy(monkeypatch):
    """Route the cycle to one fake strategy and one qualifying candidate, so each test controls the outcome."""
    state = {"entry": True, "exit": False}

    def fake(df):
        return _fire_on_last_bar(df, entry=state["entry"], exit_=state["exit"])

    monkeypatch.setitem(agent.STRATEGIES, "fake", fake)
    cand = Candidate("SPY", "fake", Metrics(0.1, 1.5, -0.05, 10, 0.6))
    monkeypatch.setattr(agent, "evaluate_all", lambda bars, s: [cand])
    monkeypatch.setattr(agent, "qualifying_best", lambda candidates, s: {"SPY": cand})
    return state


def _bars():
    return {"SPY": flat_bars(60)}


def test_market_closed_does_nothing(stub_strategy, capsys):
    broker = FakeBroker(market_open=False)
    report = agent.run_cycle(S, broker, _bars(), execute=True)
    assert report.skipped == "market_closed"
    assert broker.bought == [] and broker.closed == []


def test_fresh_entry_submits_one_bracket_buy(stub_strategy):
    broker = FakeBroker()
    report = agent.run_cycle(S, broker, _bars(), execute=True)
    assert len(broker.bought) == 1
    symbol, qty, client_id = broker.bought[0]
    assert symbol == "SPY" and qty > 0
    assert client_id.startswith("SPY-fake-")
    assert report.actions[0]["action"] == "buy"


def test_dry_run_places_no_orders_but_reports_the_decision(stub_strategy, capsys):
    broker = FakeBroker()
    report = agent.run_cycle(S, broker, _bars(), execute=False)
    assert broker.bought == []
    assert len(report.actions) == 1
    buy_lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"buy"' in line]
    assert buy_lines and buy_lines[0]["executed"] is False


def test_no_duplicate_order_for_the_same_bar(stub_strategy, capsys):
    broker = FakeBroker(recent_order=True)
    agent.run_cycle(S, broker, _bars(), execute=True)
    assert broker.bought == []
    assert '"already_traded_this_bar"' in capsys.readouterr().out


def test_already_held_symbol_is_not_bought_again(stub_strategy):
    broker = FakeBroker(positions={"SPY": 10})
    agent.run_cycle(S, broker, _bars(), execute=True)
    assert broker.bought == []


def test_exit_signal_closes_a_held_position(stub_strategy):
    stub_strategy["entry"], stub_strategy["exit"] = False, True
    broker = FakeBroker(positions={"SPY": 10})
    agent.run_cycle(S, broker, _bars(), execute=True)
    assert broker.closed == ["SPY"]


def test_daily_loss_halt_blocks_entries_but_allows_exits(stub_strategy):
    stub_strategy["exit"] = True
    broker = FakeBroker(equity=95_000.0, last_equity=100_000.0, positions={"SPY": 10})
    report = agent.run_cycle(S, broker, _bars(), execute=True)
    assert report.halted is True
    assert broker.bought == []
    assert broker.closed == ["SPY"]


def test_position_limit_blocks_new_entries(stub_strategy):
    broker = FakeBroker(positions={"AAA": 1, "BBB": 1, "CCC": 1})
    agent.run_cycle(S, broker, _bars(), execute=True)
    assert broker.bought == []
