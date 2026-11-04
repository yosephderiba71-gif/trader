"""SDT² London breakout simulator, measured in R.

Rules:
1. Levels = previous calendar day's high and low (PDH/PDL), using the last
   available prior day when the immediately preceding day has no bars.
2. Session = 04:00-11:00 UTC. Bars are labelled by their open time; the last
   session bar opens at 10:55 and closes at 11:00.
3. Entry = the first 5-minute bar in the session that closes above PDH (long)
   or below PDL (short). The fill is the next bar's open.
4. Stop = the opposite side of the signal candle. Target = 2R.
5. After every closed candle the stop moves to that candle's low (long) or high
   (short), and never moves against the trade.
6. Exit at the first of: stop, target, or the 11:00 UTC session close.
7. One trade per day.

Optional fixed variants can filter the prior-day range, filter by a 20-day
close trend, disable trailing, or delay trailing until +1R.

Conservative fills: if a bar touches both stop and target, the stop is assumed
first. Costs are deducted as a fixed R amount per trade.

Read-only research. This module never imports a broker or places orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UTC = timezone.utc
OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class SdtConfig:
    session_start: time = time(4, 0)
    session_end: time = time(11, 0)
    target_r: float = 2.0
    cost_r: float = 0.05
    max_trades_per_day: int = 1
    bar_minutes: int = 5
    prior_day_lookback: int = 7
    range_filter: bool = False
    range_min_pct: float = 0.30
    range_max_pct: float = 1.60
    trend_filter: bool = False
    trend_lookback: int = 20
    trail_mode: str = "always"

    def __post_init__(self) -> None:
        if self.trail_mode not in {"always", "never", "after_1r"}:
            raise ValueError(f"unsupported trail_mode: {self.trail_mode}")


@dataclass(frozen=True)
class SdtTrade:
    day: str
    side: str
    entry_time: str
    exit_time: str
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
    trailed: bool


@dataclass
class SdtResult:
    trades: list[SdtTrade] = field(default_factory=list)
    sessions: int = 0
    first: str | None = None
    last: str | None = None
    data_source: str = "unknown"
    config: SdtConfig = field(default_factory=SdtConfig)

    def metrics(self) -> dict[str, Any]:
        return summarize(self.trades, sessions=self.sessions)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "SDT2 London breakout",
            "data_source": self.data_source,
            "settings": config_dict(self.config),
            "metrics": self.metrics(),
            "first": self.first,
            "last": self.last,
            "trades": [asdict(trade) for trade in self.trades],
        }


def config_dict(config: SdtConfig) -> dict[str, Any]:
    return {
        "session_start": config.session_start.strftime("%H:%M"),
        "session_end": config.session_end.strftime("%H:%M"),
        "target_r": config.target_r,
        "cost_r": config.cost_r,
        "max_trades_per_day": config.max_trades_per_day,
        "bar_minutes": config.bar_minutes,
        "prior_day_lookback": config.prior_day_lookback,
        "range_filter": config.range_filter,
        "range_min_pct": config.range_min_pct,
        "range_max_pct": config.range_max_pct,
        "trend_filter": config.trend_filter,
        "trend_lookback": config.trend_lookback,
        "trail_mode": config.trail_mode,
    }


def variant_config(name: str) -> SdtConfig:
    """Return one fixed variant; variants are intentionally not combined."""
    key = name.strip().upper()
    if key == "A":
        return SdtConfig()
    if key == "B":
        return SdtConfig(range_filter=True)
    if key == "C":
        return SdtConfig(trend_filter=True)
    if key == "D":
        return SdtConfig(trail_mode="never")
    if key == "E":
        return SdtConfig(trail_mode="after_1r")
    raise ValueError(f"unknown variant: {name}")


def load_bars(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    lower = {str(column).strip().lower(): column for column in frame.columns}
    dt_key = next(
        (key for key in ("datetime", "timestamp", "date_time", "date", "time") if key in lower),
        None,
    )
    if dt_key is None:
        dt_key = str(frame.columns[0]).strip().lower()
    if dt_key not in lower:
        raise ValueError(f"{path} needs a datetime column")
    parsed = _parse_timestamps(frame[lower[dt_key]])
    frame = frame.loc[parsed.notna()].copy()
    frame.index = parsed.loc[parsed.notna()]
    return normalize_bars(frame)


def _parse_timestamps(raw: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all():
        values = numeric.astype("int64")
        magnitude = int(values.abs().max())
        if magnitude >= 10**17:
            unit = "ns"
        elif magnitude >= 10**14:
            unit = "us"
        elif magnitude >= 10**11:
            unit = "ms"
        elif magnitude >= 10**9:
            unit = "s"
        elif magnitude >= 10**7:
            return pd.to_datetime(values.astype(str), format="%Y%m%d", utc=True, errors="coerce")
        else:
            unit = "s"
        return pd.to_datetime(values, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(raw, utc=True, errors="coerce")


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("bars are empty")
    lower = {str(column).strip().lower(): column for column in frame.columns}
    missing = [column for column in ("open", "high", "low", "close") if column not in lower]
    if missing:
        raise ValueError(f"bars need columns {missing}")
    out = pd.DataFrame(
        {
            column: pd.to_numeric(frame[lower[column]], errors="coerce")
            for column in ("open", "high", "low", "close")
        }
    )
    if "volume" in lower:
        out["volume"] = pd.to_numeric(frame[lower["volume"]], errors="coerce").fillna(0.0)
    else:
        out["volume"] = 0.0
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index, utc=True, errors="coerce")
    else:
        index = pd.to_datetime(index, utc=True, errors="coerce")
    out.index = index
    out = out.loc[~out.index.isna()]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError("bars are empty after cleaning")
    return out[OHLCV]


def clean_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove flat padding and bars outside the weekend market closure.

    Flat bars are the feed's padding for closed periods. The weekend filter
    additionally removes any non-flat bars on Saturday, before Sunday 22:00
    UTC, or at/after Friday 22:00 UTC. Holiday closures are represented by
    flat padding in this feed and are removed by the flat-bar filter.
    """
    out = normalize_bars(frame)
    flat = (
        (out["open"] == out["high"])
        & (out["high"] == out["low"])
        & (out["low"] == out["close"])
    )
    out = out.loc[~flat]
    if out.empty:
        raise ValueError("bars are empty after dropping flat padding")
    minute_of_day = out.index.hour * 60 + out.index.minute
    weekday = out.index.day_of_week
    closed = (
        (weekday == 5)
        | ((weekday == 6) & (minute_of_day < 22 * 60))
        | ((weekday == 4) & (minute_of_day >= 22 * 60))
    )
    out = out.loc[~closed]
    if out.empty:
        raise ValueError("bars are empty after dropping closed-market periods")
    return out


def run_sdt(
    bars: pd.DataFrame,
    config: SdtConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    data_source: str = "unknown",
    clean: bool = True,
) -> SdtResult:
    config = config or SdtConfig()
    if config.max_trades_per_day != 1:
        raise ValueError("SDT2 is fixed at one trade per day")
    frame = clean_bars(bars) if clean else normalize_bars(bars)
    daily = {day: group for day, group in frame.groupby(frame.index.date)}
    clock = frame.index.time
    session = frame[(clock >= config.session_start) & (clock < config.session_end)]
    sessions = session.groupby(session.index.date)

    trades: list[SdtTrade] = []
    days = sorted(sessions.groups)
    for day in days:
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        prior = _prior_day(
            daily,
            day,
            config.prior_day_lookback,
            skip_sundays=clean,
        )
        if prior is None or prior.empty:
            continue
        pdh = float(prior["high"].max())
        pdl = float(prior["low"].min())
        if pdh <= pdl:
            continue
        prior_close = float(prior["close"].iloc[-1])
        if config.range_filter:
            if prior_close <= 0:
                continue
            range_pct = (pdh - pdl) / prior_close * 100.0
            if range_pct < config.range_min_pct or range_pct > config.range_max_pct:
                continue
        allowed_side = None
        if config.trend_filter:
            allowed_side = _trend_side(
                daily,
                prior.index[0].date(),
                config.trend_lookback,
            )
            if allowed_side is None:
                continue
        day_bars = sessions.get_group(day)
        trade = _run_day(
            str(day),
            day_bars,
            pdh,
            pdl,
            config,
            allowed_side=allowed_side,
        )
        if trade is not None:
            trades.append(trade)

    return SdtResult(
        trades=trades,
        sessions=len(days),
        first=str(days[0]) if days else None,
        last=str(days[-1]) if days else None,
        data_source=data_source,
        config=config,
    )


def _prior_day(
    daily: dict[date, pd.DataFrame],
    day: date,
    lookback: int,
    skip_sundays: bool = True,
) -> pd.DataFrame | None:
    for offset in range(1, lookback + 1):
        candidate = day - timedelta(days=offset)
        if skip_sundays and candidate.weekday() == 6:
            continue
        frame = daily.get(candidate)
        if frame is not None and not frame.empty:
            return frame
    return None


def _trend_side(
    daily: dict[date, pd.DataFrame],
    prior_day: date,
    lookback: int,
) -> str | None:
    dates = [day for day in sorted(daily) if day.weekday() != 6]
    if prior_day not in dates:
        return None
    position = dates.index(prior_day)
    if position < lookback:
        return None
    prior_close = float(daily[prior_day]["close"].iloc[-1])
    base_close = float(daily[dates[position - lookback]]["close"].iloc[-1])
    if prior_close > base_close:
        return "long"
    if prior_close < base_close:
        return "short"
    return None


def _run_day(
    day: str,
    bars: pd.DataFrame,
    pdh: float,
    pdl: float,
    config: SdtConfig,
    allowed_side: str | None = None,
) -> SdtTrade | None:
    opens = bars["open"].to_numpy(float)
    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    n = len(bars)
    if n < 2:
        return None

    for i in range(n - 1):
        if closes[i] > pdh:
            side = "long"
        elif closes[i] < pdl:
            side = "short"
        else:
            continue
        if allowed_side is not None and side != allowed_side:
            continue

        entry_bar = i + 1
        entry = float(opens[entry_bar])
        initial_stop = float(lows[i]) if side == "long" else float(highs[i])
        risk = entry - initial_stop if side == "long" else initial_stop - entry
        if risk <= 0:
            continue
        target = entry + config.target_r * risk if side == "long" else entry - config.target_r * risk
        exit_price, exit_bar, reason, final_stop = _manage(
            side,
            entry_bar,
            entry,
            initial_stop,
            target,
            opens,
            highs,
            lows,
            closes,
            bars.index,
            config,
        )
        sign = 1.0 if side == "long" else -1.0
        gross_r = sign * (exit_price - entry) / risk
        net_r = gross_r - config.cost_r
        trailed = (
            final_stop > initial_stop + 1e-12
            if side == "long"
            else final_stop < initial_stop - 1e-12
        )
        return SdtTrade(
            day=day,
            side=side,
            entry_time=str(bars.index[entry_bar]),
            exit_time=str(_bar_close(bars.index[exit_bar], config.bar_minutes)),
            entry=entry,
            initial_stop=initial_stop,
            final_stop=final_stop,
            target=target,
            exit=float(exit_price),
            reason=reason,
            risk=float(risk),
            gross_r=float(gross_r),
            cost_r=config.cost_r,
            net_r=float(net_r),
            bars_held=exit_bar - entry_bar + 1,
            trailed=bool(trailed),
        )
    return None


def _manage(
    side: str,
    entry_bar: int,
    entry: float,
    stop: float,
    target: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    index: pd.DatetimeIndex,
    config: SdtConfig,
) -> tuple[float, int, str, float]:
    long = side == "long"
    n = len(closes)
    initial_stop = stop
    risk = abs(entry - initial_stop)
    armed = config.trail_mode == "always"
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

        if config.trail_mode == "after_1r" and not armed:
            if long and highs[j] >= entry + risk:
                armed = True
            elif not long and lows[j] <= entry - risk:
                armed = True
        if config.trail_mode == "always" or armed:
            stop = trail_stop(side, stop, lows[j], highs[j])

    return float(closes[-1]), n - 1, "data_end", stop


def trail_stop(side: str, current: float, low: float, high: float) -> float:
    if side == "long":
        return max(float(current), float(low))
    return min(float(current), float(high))


def _bar_close(timestamp: pd.Timestamp, bar_minutes: int) -> pd.Timestamp:
    return pd.Timestamp(timestamp) + pd.Timedelta(minutes=bar_minutes)


def _session_close(first_bar: pd.Timestamp, config: SdtConfig) -> pd.Timestamp:
    day = pd.Timestamp(first_bar).date()
    return pd.Timestamp(datetime.combine(day, config.session_end, tzinfo=UTC))


def summarize(trades: list[SdtTrade], sessions: int | None = None) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "sessions": sessions or 0,
            "total_r": 0.0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "trailed_exits": 0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
            "data_end_exits": 0,
            "small_sample": True,
        }
    rs = np.array([trade.net_r for trade in trades], float)
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    cumulative = np.cumsum(rs)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    reasons = [trade.reason for trade in trades]
    return {
        "trades": len(rs),
        "sessions": sessions or len({trade.day for trade in trades}),
        "total_r": float(rs.sum()),
        "win_rate": float((rs > 0).mean()),
        "expectancy_r": float(rs.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size else None,
        "max_drawdown_r": float((cumulative - peak).min()),
        "avg_win_r": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_r": float(losses.mean()) if losses.size else 0.0,
        "long_trades": sum(trade.side == "long" for trade in trades),
        "short_trades": sum(trade.side == "short" for trade in trades),
        "trailed_exits": sum(trade.trailed for trade in trades),
        "target_exits": sum(reason in {"target", "target_gap"} for reason in reasons),
        "stop_exits": sum(reason in {"stop", "stop_gap"} for reason in reasons),
        "time_exits": sum(reason == "time" for reason in reasons),
        "data_end_exits": sum(reason == "data_end" for reason in reasons),
        "small_sample": len(rs) < 30,
    }


def make_synthetic_bars(days: int = 90, seed: int = 7, start: date = date(2026, 1, 1)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    begin = datetime.combine(start, time(0, 0), tzinfo=UTC)
    index = pd.date_range(begin, periods=days * 288, freq="5min")
    n = len(index)
    hours = np.array([timestamp.hour + timestamp.minute / 60.0 for timestamp in index])
    london = (hours >= 4.0) & (hours < 11.0)
    day_numbers = np.repeat(np.arange(days), 288)
    direction = np.where(day_numbers % 2 == 0, 1.0, -1.0)
    returns = rng.normal(0.0, 0.00035, n)
    returns[london] += 0.00015 * direction[london]
    close = 2000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[2000.0, close[:-1]]
    spread = np.abs(rng.normal(0.0, 0.00025, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
