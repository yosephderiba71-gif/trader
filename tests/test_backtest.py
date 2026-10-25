import numpy as np
import pandas as pd

from tests.helpers import flat_bars
from trader.backtest import compute_metrics, run_backtest
from trader.config import Settings
from trader.strategies import Signals, atr

NO_COSTS = Settings(slippage_bps=0.0, commission_bps=0.0, atr_stop_mult=2.0, atr_target_mult=3.0)
# ATR needs 14 bars of warm-up, so every test signals after bar 20.
ENTRY, EXIT = 20, 25


def _signals(n, entry_at=(), exit_at=()):
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    entries[list(entry_at)] = True
    exits[list(exit_at)] = True
    idx = flat_bars(n).index
    return Signals(pd.Series(entries, idx), pd.Series(exits, idx))


def test_signal_fills_on_the_next_bar_open():
    df = flat_bars(30)
    df.iloc[ENTRY + 1, df.columns.get_loc("open")] = 101.0
    df.iloc[EXIT + 1, df.columns.get_loc("open")] = 103.0
    result = run_backtest(df, _signals(30, entry_at=[ENTRY], exit_at=[EXIT]), NO_COSTS)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.reason == "signal"
    assert np.isclose(trade.ret, 103.0 / 101.0 - 1)


def test_signal_on_last_bar_cannot_fill():
    result = run_backtest(flat_bars(30), _signals(30, entry_at=[29]), NO_COSTS)
    assert result.trades == []


def test_stop_loss_fills_at_the_stop_level():
    df = flat_bars(30)
    df.iloc[ENTRY + 2, df.columns.get_loc("low")] = 90.0  # a bar that pierces any reasonable stop
    result = run_backtest(df, _signals(30, entry_at=[ENTRY]), NO_COSTS)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.reason == "stop"
    entry_fill = float(df["open"].iloc[ENTRY + 1])
    expected_stop = entry_fill - 2.0 * float(atr(df).iloc[ENTRY])
    assert np.isclose(trade.ret, expected_stop / entry_fill - 1)
    assert trade.ret < 0


def test_stop_wins_when_a_bar_touches_both_stop_and_target():
    df = flat_bars(30)
    df.iloc[ENTRY + 2, df.columns.get_loc("low")] = 90.0
    df.iloc[ENTRY + 2, df.columns.get_loc("high")] = 130.0
    result = run_backtest(df, _signals(30, entry_at=[ENTRY]), NO_COSTS)
    assert result.trades[0].reason == "stop"


def test_costs_reduce_return():
    df = flat_bars(30)
    df.iloc[ENTRY + 1, df.columns.get_loc("open")] = 101.0
    df.iloc[EXIT + 1, df.columns.get_loc("open")] = 103.0
    signals = _signals(30, entry_at=[ENTRY], exit_at=[EXIT])
    free = run_backtest(df, signals, NO_COSTS).trades[0].ret
    costly = run_backtest(df, signals, Settings(slippage_bps=10, commission_bps=5)).trades[0].ret
    assert costly < free


def test_open_position_is_closed_at_end_of_data():
    result = run_backtest(flat_bars(30), _signals(30, entry_at=[ENTRY]), NO_COSTS)
    assert result.trades[-1].reason == "end_of_data"


def test_flat_curve_has_zero_sharpe_and_drawdown():
    curve = pd.Series(np.ones(50))
    metrics = compute_metrics(curve, [], bars_per_year=1000)
    assert metrics.sharpe == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.trades == 0
