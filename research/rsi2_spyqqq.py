"""RSI(2) mean reversion on SPY and QQQ with the frozen rules.

The rules are frozen in ``research/rsi2_rules.md`` and must not change after
any result is seen. This module is simulator-only: it never imports a broker
and never places an order. No parameter is tuned; the in-sample split exists
only as a reporting requirement, not as a tuning constraint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trader.strategies import rsi as wilder_rsi
from trader.strategies import sma

BASE_COST_BPS = 5.0
STRESS_COST_BPS = 10.0
COST_BPS = (BASE_COST_BPS, STRESS_COST_BPS)
TRAIN_FRACTION = 0.70
HALF_FRACTION = 0.50
RSI_PERIOD = 2
SMA_ENTRY_PERIOD = 200
SMA_EXIT_PERIOD = 5
ENTRY_RSI = 10.0
MAX_HOLDING_DAYS = 10
BASELINE_RUNS = 1000
BASELINE_PERCENTILE = 95.0


@dataclass(frozen=True)
class Rsi2Trade:
    symbol: str
    entry_index: int
    entry_fill_index: int
    exit_index: int | None
    exit_fill_index: int | None
    entry_open: float
    exit_open: float
    held_days: int
    exit_reason: str | None
    ret_gross_pct: float
    ret_net_base_pct: float
    ret_net_stress_pct: float


def load_daily_bars(
    symbol: str,
    start: str | None = "2000-01-01",
    end: str | None = "2026-11-05",
) -> pd.DataFrame:
    """Fetch dividend/split-adjusted daily OHLC from Yahoo Finance."""
    import yfinance as yf

    frame = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
    if frame is None or frame.empty:
        raise ValueError(f"no daily bars for {symbol}")
    prepared = _prepare(frame)
    if end is not None:
        prepared = prepared.loc[prepared.index <= pd.Timestamp(end, tz="UTC")]
    if start is not None:
        prepared = prepared.loc[prepared.index >= pd.Timestamp(start, tz="UTC")]
    return prepared


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    missing = [column for column in ("open", "high", "low", "close") if column not in out.columns]
    if missing:
        raise ValueError(f"bars need columns {missing}")
    out = out.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    index = pd.DatetimeIndex(out.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    out.index = index
    out = out.loc[~out.index.isna()]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    return out


def _net_return_pct(buy_open: float, sell_open: float, cost_bps: float) -> float:
    cost = cost_bps / 10000.0
    buy_fill = buy_open * (1.0 + cost)
    sell_fill = sell_open * (1.0 - cost)
    return float((sell_fill / buy_fill - 1.0) * 100.0)


def simulate(bars: pd.DataFrame, symbol: str = "SPY") -> list[Rsi2Trade]:
    """Run the frozen one-position-at-a-time RSI(2) mean-reversion simulator."""
    frame = _prepare(bars)
    n = len(frame)
    if n == 0:
        return []
    opens = frame["open"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    rsi2 = wilder_rsi(frame["close"], RSI_PERIOD)
    sma200 = sma(frame["close"], SMA_ENTRY_PERIOD)
    sma5 = sma(frame["close"], SMA_EXIT_PERIOD)

    trades: list[Rsi2Trade] = []
    position: dict[str, Any] | None = None
    for i in range(n):
        if position is None:
            value = rsi2.iloc[i]
            sma_value = sma200.iloc[i]
            if pd.isna(value) or pd.isna(sma_value):
                continue
            if not (float(closes[i]) > float(sma_value) and float(value) < ENTRY_RSI):
                continue
            if i + 1 >= n:
                continue
            position = {
                "entry_index": i,
                "entry_fill_index": i + 1,
                "entry_open": float(opens[i + 1]),
            }
            continue

        entry_fill_index = int(position["entry_fill_index"])
        held_days = (i - entry_fill_index) + 1
        exit_now = held_days >= MAX_HOLDING_DAYS or bool(closes[i] > float(sma5.iloc[i]))
        if not exit_now:
            continue

        exit_fill_index = i + 1
        if exit_fill_index >= n:
            continue
        exit_open = float(opens[exit_fill_index])
        reason = "time_stop" if held_days >= MAX_HOLDING_DAYS else "sma5"
        buy_open = float(position["entry_open"])
        trades.append(
            Rsi2Trade(
                symbol=symbol,
                entry_index=int(position["entry_index"]),
                entry_fill_index=entry_fill_index,
                exit_index=int(i),
                exit_fill_index=int(exit_fill_index),
                entry_open=buy_open,
                exit_open=exit_open,
                held_days=int(held_days),
                exit_reason=reason,
                ret_gross_pct=float((exit_open / buy_open - 1.0) * 100.0),
                ret_net_base_pct=_net_return_pct(buy_open, exit_open, BASE_COST_BPS),
                ret_net_stress_pct=_net_return_pct(buy_open, exit_open, STRESS_COST_BPS),
            )
        )
        position = None

    return trades


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    return float((cumulative - peak).min())


def summarize(trades: list[Rsi2Trade], cost_bps: float) -> dict[str, Any]:
    """Summary metrics for a subset of trades at one cost assumption."""
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "expectancy_pct": 0.0,
            "profit_factor": None,
            "max_drawdown_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }
    net = np.array(
        [
            trade.ret_net_base_pct if cost_bps == BASE_COST_BPS else trade.ret_net_stress_pct
            for trade in trades
        ],
        dtype=float,
    )
    wins = net[net > 0]
    losses = net[net < 0]
    return {
        "trades": int(net.size),
        "win_rate": float((net > 0).mean()),
        "expectancy_pct": float(net.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size else None,
        "max_drawdown_pct": _max_drawdown(net),
        "avg_win_pct": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_pct": float(losses.mean()) if losses.size else 0.0,
    }


def random_baseline(
    bars: pd.DataFrame,
    entry_indices: np.ndarray,
    held_periods: np.ndarray,
    cost_bps: float = BASE_COST_BPS,
) -> dict[str, Any]:
    """1,000 random entry-day strategies holding the real holding-period mix."""
    frame = _prepare(bars)
    n = len(frame)
    opens = frame["open"].to_numpy(dtype=float)
    if n == 0 or entry_indices.size == 0 or held_periods.size == 0:
        return {"runs": 0, "baseline_pct": float("nan"), "expectancy_pct": float("nan")}

    max_held = int(max(held_periods.max(), 1))
    safe = entry_indices[entry_indices + 1 + max_held < n]
    if safe.size == 0:
        return {"runs": 0, "baseline_pct": float("nan"), "expectancy_pct": float("nan")}

    rng = np.random.default_rng(20261105)
    rets: list[float] = []
    for _ in range(BASELINE_RUNS):
        picks = rng.choice(safe, size=len(entry_indices), replace=True)
        held = rng.choice(held_periods, size=len(picks), replace=True)
        values = []
        for entry_index, holding in zip(picks, held):
            buy_open = float(opens[int(entry_index) + 1])
            sell_open = float(opens[int(entry_index) + 1 + int(holding)])
            values.append(_net_return_pct(buy_open, sell_open, cost_bps))
        rets.append(float(np.mean(values)))
    rets_array = np.array(rets, dtype=float)
    return {
        "runs": int(len(rets)),
        "baseline_pct": float(np.percentile(rets_array, BASELINE_PERCENTILE)),
        "expectancy_pct": float(rets_array.mean()),
    }
