"""Walk-forward testing: does picking a strategy on the past still work on the next stretch?

For each fold:
  1. Train window: score every strategy on bars it has not seen later.
  2. Choose the best training strategy that clears the gates (min trades, positive return).
  3. Test window: apply only that choice to the following bars and record the result.

Folds roll forward through history, so every test window is data the choice never saw.
Signals are causal, so scoring a window never uses bars after the window's end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Metrics, compute_metrics, run_backtest
from .config import Settings
from .strategies import STRATEGIES, Signals


@dataclass(frozen=True)
class Fold:
    train_start: int
    test_start: int
    test_end: int
    chosen: str | None  # None when no strategy passed the training gates (no trades)
    test: Metrics | None


def score_window(df: pd.DataFrame, signals: Signals, s: Settings, bars_per_year: float,
                 start: int, end: int) -> Metrics:
    """Score only bars [start, end). Bars after `end` are never read."""
    head = df.iloc[:end]
    sig = Signals(signals.entries.iloc[:end], signals.exits.iloc[:end])
    result = run_backtest(head, sig, s)
    curve = result.equity.iloc[start:end]
    curve = curve / curve.iloc[0]
    trades = [t for t in result.trades if start <= t.exit_index < end]
    return compute_metrics(curve, trades, bars_per_year)


def _passes_gates(m: Metrics, s: Settings) -> bool:
    return m.trades >= s.min_oos_trades and m.total_return > 0 and not np.isnan(m.sharpe)


def walk_forward(df: pd.DataFrame, s: Settings, bars_per_year: float,
                 train_bars: int = 500, test_bars: int = 125) -> list[Fold]:
    """Roll train/test windows across df. Test windows do not overlap."""
    if train_bars < 120 or test_bars < 1:
        raise ValueError("train_bars must be at least 120 and test_bars at least 1")
    if len(df) < train_bars + test_bars:
        raise ValueError(f"need at least {train_bars + test_bars} bars, got {len(df)}")

    all_signals = {name: make(df) for name, make in STRATEGIES.items()}
    folds: list[Fold] = []
    start = 0
    while start + train_bars + test_bars <= len(df):
        train_end = start + train_bars
        test_end = train_end + test_bars

        best_name, best_sharpe = None, -np.inf
        for name, sig in all_signals.items():
            m = score_window(df, sig, s, bars_per_year, start, train_end)
            if _passes_gates(m, s) and m.sharpe > best_sharpe:
                best_name, best_sharpe = name, m.sharpe

        test = None
        if best_name is not None:
            test = score_window(df, all_signals[best_name], s, bars_per_year, train_end, test_end)
        folds.append(Fold(start, train_end, test_end, best_name, test))
        start += test_bars  # step by one test window so test windows never overlap
    return folds


def summarize(folds: list[Fold]) -> dict:
    """Compound the test returns of the folds where a strategy was chosen."""
    traded = [f for f in folds if f.test is not None]
    returns = [f.test.total_return for f in traded]
    compounded = float(np.prod([1 + r for r in returns]) - 1) if returns else 0.0
    return {
        "folds": len(folds),
        "folds_traded": len(traded),
        "compounded_test_return": compounded,
        "mean_test_return": float(np.mean(returns)) if returns else 0.0,
        "share_positive": float(np.mean([r > 0 for r in returns])) if returns else 0.0,
    }
