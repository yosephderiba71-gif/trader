import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_bars
from trader.strategies import STRATEGIES, cross_above, cross_below, rsi


def test_cross_above_fires_once_on_the_crossing_bar():
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert cross_above(a, 2.5).tolist() == [False, False, True, False]


def test_cross_below_fires_once_on_the_crossing_bar():
    a = pd.Series([4.0, 3.0, 2.0, 1.0])
    assert cross_below(a, 2.5).tolist() == [False, False, True, False]


def test_rsi_stays_in_bounds_after_warmup():
    value = rsi(make_bars(300)["close"]).dropna()
    assert ((value >= 0) & (value <= 100)).all()


def test_ma_crossover_enters_after_a_downtrend_turns_up():
    down = np.linspace(120, 80, 100)
    up = np.linspace(80, 140, 100)
    close = np.r_[down, up]
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": np.full(200, 1e4)},
        index=pd.date_range("2025-01-02", periods=200, freq="h", tz="UTC"),
    )
    signals = STRATEGIES["ma_crossover"](df)
    assert signals.entries.any()
    assert signals.entries[signals.entries].index.min() > df.index[100]


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_returns_boolean_series_aligned_to_bars(name):
    df = make_bars(300)
    signals = STRATEGIES[name](df)
    for series in (signals.entries, signals.exits):
        assert series.dtype == bool
        assert series.index.equals(df.index)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_is_causal(name):
    """Values before bar k must not change when later bars are removed."""
    df = make_bars(400, seed=3)
    k = 300
    full = STRATEGIES[name](df)
    truncated = STRATEGIES[name](df.iloc[:k])
    assert full.entries.iloc[:k].equals(truncated.entries)
    assert full.exits.iloc[:k].equals(truncated.exits)
