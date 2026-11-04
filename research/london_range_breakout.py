"""Range-breakout and failed-break simulators for XAUUSD.

Variants F, G, and H are fixed research configurations. This module is
simulator-only and never imports a broker or places orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Any

import pandas as pd

from research.london_sdt import UTC, clean_bars, normalize_bars, summarize


@dataclass(frozen=True)
class RangeConfig:
    name: str
    range_start: time
    range_end: time
    entry_start: time
    entry_end: time
    mode: str
    target_r: float = 2.0
    target_kind: str = "r"
    cost_r: float = 0.05
    bar_minutes: int = 5
    failed_lookback: int = 3
    max_trades_per_day: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"breakout", "failed_break"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.target_kind not in {"r", "midpoint"}:
            raise ValueError(f"unsupported target_kind: {self.target_kind}")
        if self.target_r <= 0:
            raise ValueError("target_r must be positive")
        if self.failed_lookback < 1:
            raise ValueError("failed_lookback must be positive")
        if self.max_trades_per_day != 1:
            raise ValueError("range variants are fixed at one trade per day")
        if not (
            self.range_start
            < self.range_end
            <= self.entry_start
            < self.entry_end
        ):
            raise ValueError("range and entry windows are invalid")


@dataclass(frozen=True)
class RangeTrade:
    day: str
    variant: str
    side: str
    entry_time: str
    exit_time: str
    range_high: float
    range_low: float
    entry: float
    initial_stop: float
    final_stop: float
    target: float
    exit: float
    reason: str
    risk: float
    gross_r: float
    cost_r: float
    net_r: float
    bars_held: int
    trailed: bool = False


@dataclass
class RangeResult:
    config: RangeConfig
    trades: list[RangeTrade] = field(default_factory=list)
    sessions: int = 0
    first: str | None = None
    last: str | None = None
    data_source: str = "unknown"

    def metrics(self) -> dict[str, Any]:
        return summarize(self.trades, sessions=self.sessions)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": f"Range variant {self.config.name}",
            "data_source": self.data_source,
            "settings": config_dict(self.config),
            "metrics": self.metrics(),
            "first": self.first,
            "last": self.last,
            "trades": [asdict(trade) for trade in self.trades],
        }


def range_variant_config(name: str, cost_r: float = 0.05) -> RangeConfig:
    key = name.strip().upper()
    if key == "F":
        return RangeConfig(
            "F",
            time(0, 0),
            time(7, 0),
            time(7, 0),
            time(11, 0),
            "breakout",
            cost_r=cost_r,
        )
    if key == "G":
        return RangeConfig(
            "G",
            time(7, 0),
            time(8, 0),
            time(8, 0),
            time(11, 0),
            "breakout",
            cost_r=cost_r,
        )
    if key == "H":
        return RangeConfig(
            "H",
            time(0, 0),
            time(7, 0),
            time(7, 0),
            time(11, 0),
            "failed_break",
            target_kind="midpoint",
            cost_r=cost_r,
        )
    raise ValueError(f"unknown range variant: {name}")


def config_dict(config: RangeConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "range_start": config.range_start.strftime("%H:%M"),
        "range_end": config.range_end.strftime("%H:%M"),
        "entry_start": config.entry_start.strftime("%H:%M"),
        "entry_end": config.entry_end.strftime("%H:%M"),
        "mode": config.mode,
        "target_r": config.target_r,
        "target_kind": config.target_kind,
        "cost_r": config.cost_r,
        "bar_minutes": config.bar_minutes,
        "failed_lookback": config.failed_lookback,
        "max_trades_per_day": config.max_trades_per_day,
    }


def run_range_variant(
    bars: pd.DataFrame,
    config: RangeConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    data_source: str = "unknown",
    clean: bool = True,
) -> RangeResult:
    config = config or range_variant_config("F")
    frame = clean_bars(bars) if clean else normalize_bars(bars)
    daily = {day: group for day, group in frame.groupby(frame.index.date)}
    days = sorted(daily)
    trades: list[RangeTrade] = []
    for day in days:
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        day_bars = daily[day]
        clock = day_bars.index.time
        range_bars = day_bars[
            (clock >= config.range_start) & (clock < config.range_end)
        ]
        if range_bars.empty:
            continue
        range_high = float(range_bars["high"].max())
        range_low = float(range_bars["low"].min())
        if range_high <= range_low:
            continue
        entry_bars = day_bars[
            (clock >= config.entry_start) & (clock < config.entry_end)
        ]
        if len(entry_bars) < 2:
            continue
        if config.mode == "breakout":
            trade = _run_breakout_day(
                str(day),
                entry_bars,
                range_high,
                range_low,
                config,
            )
        else:
            trade = _run_failed_break_day(
                str(day),
                entry_bars,
                range_high,
                range_low,
                config,
            )
        if trade is not None:
            trades.append(trade)
    return RangeResult(
        config=config,
        trades=trades,
        sessions=len(days),
        first=str(days[0]) if days else None,
        last=str(days[-1]) if days else None,
        data_source=data_source,
    )


def _run_breakout_day(
    day: str,
    bars: pd.DataFrame,
    range_high: float,
    range_low: float,
    config: RangeConfig,
) -> RangeTrade | None:
    opens = bars["open"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    n = len(bars)
    for i in range(n - 1):
        if closes[i] > range_high:
            side = "long"
            stop = range_low
        elif closes[i] < range_low:
            side = "short"
            stop = range_high
        else:
            continue
        entry_bar = i + 1
        entry = float(opens[entry_bar])
        risk = entry - stop if side == "long" else stop - entry
        if risk <= 0:
            continue
        target = (
            entry + config.target_r * risk
            if side == "long"
            else entry - config.target_r * risk
        )
        return _finish_trade(
            day,
            side,
            entry_bar,
            entry,
            stop,
            target,
            bars,
            config,
            range_high,
            range_low,
        )
    return None


def _run_failed_break_day(
    day: str,
    bars: pd.DataFrame,
    range_high: float,
    range_low: float,
    config: RangeConfig,
) -> RangeTrade | None:
    opens = bars["open"].to_numpy(float)
    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    n = len(bars)
    midpoint = (range_high + range_low) / 2.0
    for i in range(n - 1):
        if closes[i] > range_high:
            side = "short"
        elif closes[i] < range_low:
            side = "long"
        else:
            continue
        max_j = min(i + config.failed_lookback, n - 2)
        for j in range(i + 1, max_j + 1):
            if not (range_low <= closes[j] <= range_high):
                continue
            if side == "short":
                stop = float(highs[i : j + 1].max())
            else:
                stop = float(lows[i : j + 1].min())
            entry_bar = j + 1
            entry = float(opens[entry_bar])
            target = midpoint
            if side == "short":
                if stop <= entry or target >= entry:
                    continue
            else:
                if stop >= entry or target <= entry:
                    continue
            return _finish_trade(
                day,
                side,
                entry_bar,
                entry,
                stop,
                target,
                bars,
                config,
                range_high,
                range_low,
            )
    return None


def _finish_trade(
    day: str,
    side: str,
    entry_bar: int,
    entry: float,
    stop: float,
    target: float,
    bars: pd.DataFrame,
    config: RangeConfig,
    range_high: float,
    range_low: float,
) -> RangeTrade:
    opens = bars["open"].to_numpy(float)
    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    exit_price, exit_bar, reason, final_stop = _manage(
        side,
        entry_bar,
        entry,
        stop,
        target,
        opens,
        highs,
        lows,
        closes,
        bars.index,
        config,
    )
    sign = 1.0 if side == "long" else -1.0
    risk = entry - stop if side == "long" else stop - entry
    gross_r = sign * (exit_price - entry) / risk
    net_r = gross_r - config.cost_r
    return RangeTrade(
        day=day,
        variant=config.name,
        side=side,
        entry_time=str(bars.index[entry_bar]),
        exit_time=str(_bar_close(bars.index[exit_bar], config.bar_minutes)),
        range_high=range_high,
        range_low=range_low,
        entry=entry,
        initial_stop=stop,
        final_stop=final_stop,
        target=target,
        exit=float(exit_price),
        reason=reason,
        risk=float(risk),
        gross_r=float(gross_r),
        cost_r=config.cost_r,
        net_r=float(net_r),
        bars_held=exit_bar - entry_bar + 1,
    )


def _manage(
    side: str,
    entry_bar: int,
    entry: float,
    stop: float,
    target: float,
    opens,
    highs,
    lows,
    closes,
    index: pd.DatetimeIndex,
    config: RangeConfig,
) -> tuple[float, int, str, float]:
    long = side == "long"
    n = len(closes)
    for j in range(entry_bar, n):
        if long:
            if opens[j] <= stop:
                return float(opens[j]), j, "stop_gap", stop
            if lows[j] <= stop and highs[j] >= target:
                return stop, j, "stop", stop
            if lows[j] <= stop:
                return stop, j, "stop", stop
            if opens[j] >= target:
                return float(opens[j]), j, "target_gap", stop
            if highs[j] >= target:
                return target, j, "target", stop
        else:
            if opens[j] >= stop:
                return float(opens[j]), j, "stop_gap", stop
            if highs[j] >= stop and lows[j] <= target:
                return stop, j, "stop", stop
            if highs[j] >= stop:
                return stop, j, "stop", stop
            if opens[j] <= target:
                return float(opens[j]), j, "target_gap", stop
            if lows[j] <= target:
                return target, j, "target", stop
        if _bar_close(index[j], config.bar_minutes) >= _session_close(index[0], config):
            return float(closes[j]), j, "time", stop
        if j == n - 1:
            return float(closes[j]), j, "data_end", stop
    return float(closes[-1]), n - 1, "data_end", stop


def _bar_close(timestamp: pd.Timestamp, bar_minutes: int) -> pd.Timestamp:
    return pd.Timestamp(timestamp) + pd.Timedelta(minutes=bar_minutes)


def _session_close(first_bar: pd.Timestamp, config: RangeConfig) -> pd.Timestamp:
    day = pd.Timestamp(first_bar).date()
    return pd.Timestamp(datetime.combine(day, config.entry_end, tzinfo=UTC))
