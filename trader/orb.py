"""Opening-range breakout with a break-of-structure trigger, measured in R.

Rules, adapted from the SDT² write-up for US ETFs (the London open becomes the US cash open):
1. Opening range = high and low of the first bar of the session.
2. Observation window = the first `window_bars` bars. No entries inside it.
3. Entry = the first bar after the window that closes beyond the opening range
   (above the high for a long, below the low for a short). Filled at the NEXT bar's open,
   so the decision never uses a price that was not yet known.
4. Stop = the opposite side of the opening range. Target = `target_r` times the risk.
5. Trailing stop: after each bar closes, the stop moves to that bar's low (long) or high
   (short), and never moves against the trade.
6. Exit at the first of: stop, target, `max_hold_bars` after entry, or the last bar of the session.
7. One trade per session.

Conservative fills: if one bar touches both stop and target, the stop is assumed first.
Results are in R (multiples of the initial risk), less `cost_r` per trade for costs and slippage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrbConfig:
    session_open: time = time(9, 30)
    session_close: time = time(15, 55)
    tz: str = "America/New_York"
    window_bars: int = 6        # first 30 minutes of 5-minute bars
    target_r: float = 2.0       # reward-to-risk 1:2
    max_hold_bars: int = 6      # about 30 minutes after entry, as the write-up's scalp suggests
    cost_r: float = 0.05        # per-trade costs and slippage, in R
    trail: bool = True          # True: stop follows each bar's low/high. False: stop stays at the range edge


@dataclass(frozen=True)
class OrbTrade:
    day: str
    side: str
    entry: float
    exit: float
    r: float
    reason: str


def _sessions(df: pd.DataFrame, cfg: OrbConfig):
    local = df.copy()
    local.index = local.index.tz_convert(cfg.tz)
    clock = local.index.time
    regular = local[(clock >= cfg.session_open) & (clock <= cfg.session_close)]
    return regular.groupby(regular.index.date)


def run_orb(df: pd.DataFrame, cfg: OrbConfig = OrbConfig()) -> list[OrbTrade]:
    """Run the rules over 5-minute (or any fixed-interval) OHLC bars with a tz-aware index."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"bars need columns {sorted(required)}")
    trades: list[OrbTrade] = []
    for day, bars in _sessions(df, cfg):
        trade = _run_day(str(day), bars, cfg)
        if trade is not None:
            trades.append(trade)
    return trades


def _run_day(day: str, bars: pd.DataFrame, cfg: OrbConfig) -> OrbTrade | None:
    opens = bars["open"].to_numpy(float)
    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    n = len(bars)
    if n < cfg.window_bars + 2:
        return None

    range_high, range_low = highs[0], lows[0]
    for i in range(cfg.window_bars, n - 1):  # signal on bar i, fill at bar i+1's open
        if closes[i] > range_high:
            side = "long"
        elif closes[i] < range_low:
            side = "short"
        else:
            continue

        entry_bar = i + 1
        entry = opens[entry_bar]
        if side == "long":
            stop = range_low
            risk = entry - stop
            target = entry + cfg.target_r * risk
        else:
            stop = range_high
            risk = stop - entry
            target = entry - cfg.target_r * risk
        if risk <= 0:
            return None  # the entry is already beyond the stop: no usable trade today

        exit_price, reason = _manage(side, entry_bar, entry, stop, target, opens, highs, lows, closes, cfg)
        sign = 1.0 if side == "long" else -1.0
        r = sign * (exit_price - entry) / risk - cfg.cost_r
        return OrbTrade(day, side, float(entry), float(exit_price), float(r), reason)
    return None


def _manage(side, entry_bar, entry, stop, target, opens, highs, lows, closes, cfg):
    n = len(closes)
    long = side == "long"
    for j in range(entry_bar, n):
        # 1) Stop and target, using the stop as it stood at the start of this bar.
        if long:
            if opens[j] <= stop:
                return opens[j], "stop_gap"
            if lows[j] <= stop:
                return stop, "stop"
            if opens[j] >= target:
                return opens[j], "target_gap"
            if highs[j] >= target:
                return target, "target"
        else:
            if opens[j] >= stop:
                return opens[j], "stop_gap"
            if highs[j] >= stop:
                return stop, "stop"
            if opens[j] <= target:
                return opens[j], "target_gap"
            if lows[j] <= target:
                return target, "target"

        # 2) Time exit: max hold reached, or the session's last bar.
        if j == n - 1 or (j - entry_bar) >= cfg.max_hold_bars - 1:
            return closes[j], "time"

        # 3) Trail on this bar's close. The next bar sees the new stop, never an earlier one.
        if not cfg.trail:
            continue
        if long:
            stop = max(stop, lows[j])
        else:
            stop = min(stop, highs[j])
    return closes[-1], "time"


def summarize(trades: list[OrbTrade]) -> dict:
    """Win rate, expectancy and drawdown in R. `small_sample` below 30 trades."""
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0,
                "max_drawdown_r": 0.0, "small_sample": True}
    rs = np.array([t.r for t in trades], float)
    wins = rs[rs > 0].sum()
    losses = -rs[rs <= 0].sum()
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return {
        "trades": len(rs),
        "win_rate": float((rs > 0).mean()),
        "expectancy_r": float(rs.mean()),
        # None, not infinity: infinity is not valid JSON, and the report is written as JSON.
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "max_drawdown_r": float((cum - peak).min()),
        "small_sample": len(rs) < 30,
    }
