"""Signal generators.

Each strategy returns two boolean Series aligned to the bars: entry events and exit
events, both evaluated at bar close. Every calculation is causal (the value at bar t
uses only bars up to t), which tests/test_strategies.py checks directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pandas as pd

Level = Union[pd.Series, float]


@dataclass(frozen=True)
class Signals:
    entries: pd.Series
    exits: pd.Series


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).mask((loss == 0) & gain.notna(), 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(n).mean()


def cross_above(a: pd.Series, b: Level) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift(1) <= b_prev)


def cross_below(a: pd.Series, b: Level) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a < b) & (a.shift(1) >= b_prev)


def ma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> Signals:
    f, s = sma(df["close"], fast), sma(df["close"], slow)
    return Signals(entries=cross_above(f, s), exits=cross_below(f, s))


def rsi_reversion(df: pd.DataFrame, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> Signals:
    r = rsi(df["close"], period)
    return Signals(entries=cross_above(r, oversold), exits=cross_above(r, overbought))


def bollinger_breakout(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> Signals:
    mid = sma(df["close"], n)
    upper = mid + k * df["close"].rolling(n).std()
    return Signals(entries=cross_above(df["close"], upper), exits=cross_below(df["close"], mid))


def vwap_ema(df: pd.DataFrame, ema_n: int = 9, vwap_n: int = 20, vol_n: int = 20) -> Signals:
    """Rolling VWAP (not cumulative from the start of history), with a volume filter."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["volume"]).rolling(vwap_n).sum() / df["volume"].rolling(vwap_n).sum()
    fast = ema(df["close"], ema_n)
    liquid = df["volume"] > sma(df["volume"], vol_n)
    return Signals(entries=cross_above(fast, vwap) & liquid, exits=cross_below(fast, vwap))


STRATEGIES: dict[str, Callable[[pd.DataFrame], Signals]] = {
    "ma_crossover": ma_crossover,
    "rsi_reversion": rsi_reversion,
    "bollinger_breakout": bollinger_breakout,
    "vwap_ema": vwap_ema,
}
