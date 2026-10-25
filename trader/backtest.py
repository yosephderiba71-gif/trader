"""Long-only event backtester with conservative fills.

Assumptions, chosen to avoid flattering results:
- A signal on bar t fills at the OPEN of bar t+1.
- Stop and target are ATR-based, set at entry, the same as live trading.
- If one bar touches both stop and target, the stop is assumed to hit first.
- Slippage and commission are charged on both sides of every trade.
- Each trade uses all capital. This is for ranking strategies only; live sizing is in risk.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Settings
from .strategies import Signals, atr


@dataclass(frozen=True)
class Trade:
    exit_index: int
    ret: float
    reason: str


@dataclass
class BacktestResult:
    equity: pd.Series  # mark-to-market curve, starts at 1.0
    trades: list[Trade]


@dataclass(frozen=True)
class Metrics:
    total_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float


def _ret(entry_fill: float, exit_fill: float, commission: float) -> float:
    return exit_fill * (1 - commission) / (entry_fill * (1 + commission)) - 1


def run_backtest(df: pd.DataFrame, signals: Signals, s: Settings) -> BacktestResult:
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    atr_values = atr(df).to_numpy(float)
    entries = signals.entries.to_numpy(bool)
    exits = signals.exits.to_numpy(bool)
    slip = s.slippage_bps / 10_000
    comm = s.commission_bps / 10_000

    n = len(df)
    curve = np.ones(n)
    trades: list[Trade] = []
    equity = 1.0  # realised equity multiplier
    in_pos = False
    entry_fill = stop = target = 0.0
    queued_entry = queued_exit = False
    queued_atr = np.nan

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        do_entry, do_exit, entry_atr = queued_entry, queued_exit, queued_atr

        # 1) Orders queued on bar i-1 fill at this bar's open.
        if in_pos and do_exit:
            ret = _ret(entry_fill, o * (1 - slip), comm)
            trades.append(Trade(i, ret, "signal"))
            equity *= 1 + ret
            in_pos = False
        if not in_pos and do_entry and np.isfinite(entry_atr):
            entry_fill = o * (1 + slip)
            stop = entry_fill - s.atr_stop_mult * entry_atr
            target = entry_fill + s.atr_target_mult * entry_atr
            in_pos = True

        # 2) Intrabar protection. A gap fills at the open; a touch fills at the level.
        if in_pos:
            px, reason = None, ""
            if o <= stop:
                px, reason = o, "stop_gap"
            elif l <= stop:
                px, reason = stop * (1 - slip), "stop"
            elif o >= target:
                px, reason = o, "target_gap"
            elif h >= target:
                px, reason = target * (1 - slip), "target"
            if px is not None:
                ret = _ret(entry_fill, px, comm)
                trades.append(Trade(i, ret, reason))
                equity *= 1 + ret
                in_pos = False

        # 3) Queue what bar i asks for. It fills on bar i+1.
        queued_entry, queued_exit, queued_atr = bool(entries[i]), bool(exits[i]), atr_values[i]
        curve[i] = equity * (c / entry_fill) if in_pos else equity

    if in_pos:
        ret = _ret(entry_fill, closes[-1] * (1 - slip), comm)
        trades.append(Trade(n - 1, ret, "end_of_data"))
        equity *= 1 + ret
        curve[-1] = equity

    return BacktestResult(pd.Series(curve, index=df.index, name="equity"), trades)


def compute_metrics(curve: pd.Series, trades: list[Trade], bars_per_year: float) -> Metrics:
    rets = curve.pct_change().dropna()
    sd = float(rets.std()) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean()) / sd * float(np.sqrt(bars_per_year)) if sd > 0 else 0.0
    drawdown = curve / curve.cummax() - 1
    wins = sum(1 for t in trades if t.ret > 0)
    return Metrics(
        total_return=float(curve.iloc[-1] / curve.iloc[0] - 1),
        sharpe=sharpe,
        max_drawdown=float(drawdown.min()),
        trades=len(trades),
        win_rate=wins / len(trades) if trades else 0.0,
    )


def evaluate_oos(df: pd.DataFrame, signals: Signals, s: Settings, bars_per_year: float) -> Metrics:
    """Score a strategy only on the final oos_fraction of history.

    Indicators warm up on the earlier part of the data, so the scored period starts
    with valid values.
    """
    result = run_backtest(df, signals, s)
    split = int(len(df) * (1 - s.oos_fraction))
    curve = result.equity.iloc[split:]
    curve = curve / curve.iloc[0]
    trades = [t for t in result.trades if t.exit_index >= split]
    return compute_metrics(curve, trades, bars_per_year)
