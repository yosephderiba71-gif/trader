import numpy as np
import pandas as pd
import pytest

from tests.helpers import flat_bars, make_bars
from trader.config import Settings
from trader.strategies import STRATEGIES, Signals
from trader.walk_forward import Fold, score_window, summarize, walk_forward

BPY = 252 * 6.5


def _empty_signals(df):
    idx = df.index
    return Signals(pd.Series(False, idx), pd.Series(False, idx))


def test_rejects_data_that_is_too_short():
    with pytest.raises(ValueError):
        walk_forward(make_bars(200), Settings(), BPY, train_bars=150, test_bars=100)


def test_rejects_train_window_below_indicator_warmup():
    with pytest.raises(ValueError):
        walk_forward(make_bars(400), Settings(), BPY, train_bars=100, test_bars=50)


def test_test_windows_roll_forward_without_overlap():
    folds = walk_forward(make_bars(400), Settings(), BPY, train_bars=120, test_bars=60)
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.test_start - prev.test_start == 60
        assert nxt.test_start >= prev.test_end - 60
    assert all(f.test_end <= 400 for f in folds)


def test_score_window_never_reads_bars_after_its_end():
    df = make_bars(400, seed=3)
    signals = STRATEGIES["ma_crossover"](df)
    original = score_window(df, signals, Settings(), BPY, 150, 250)

    tampered = df.copy()
    tampered.iloc[250:, tampered.columns.get_loc("close")] *= 10  # wildly different future
    tampered.iloc[250:, tampered.columns.get_loc("open")] *= 10
    assert score_window(tampered, signals, Settings(), BPY, 150, 250) == original


def test_flat_market_with_no_signals_makes_no_trades():
    df = flat_bars(300)
    m = score_window(df, _empty_signals(df), Settings(), BPY, 150, 300)
    assert m.trades == 0
    assert m.total_return == 0.0


def test_no_strategy_passing_gates_means_no_test_trade():
    folds = walk_forward(flat_bars(400), Settings(), BPY, train_bars=120, test_bars=60)
    assert folds and all(f.chosen is None and f.test is None for f in folds)
    assert summarize(folds)["folds_traded"] == 0


def test_summarize_compounds_only_traded_folds():
    class M:  # minimal stand-in for Metrics, which summarize reads via total_return
        def __init__(self, r):
            self.total_return = r

    folds = [
        Fold(0, 120, 180, "ma_crossover", M(0.10)),
        Fold(60, 180, 240, None, None),
        Fold(120, 240, 300, "rsi_reversion", M(-0.05)),
    ]
    out = summarize(folds)
    assert out["folds"] == 3
    assert out["folds_traded"] == 2
    assert np.isclose(out["compounded_test_return"], 1.10 * 0.95 - 1)
    assert np.isclose(out["share_positive"], 0.5)


def test_summarize_with_no_trades_is_zero():
    out = summarize([])
    assert out == {
        "folds": 0,
        "folds_traded": 0,
        "compounded_test_return": 0.0,
        "mean_test_return": 0.0,
        "share_positive": 0.0,
    }
