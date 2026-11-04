"""Time-of-day noise bands and session VWAP.

Formula from the paper, computed without look-ahead:

    sigma(t, slot) = mean over the previous N sessions of
                     abs(close(session, slot) / open(session, 09:30) - 1)

    upper = max(today_open, prior_close) * (1 + vm * sigma)
    lower = min(today_open, prior_close) * (1 - vm * sigma)

``slot`` is the bar's *end* time in New York minutes (10:00 -> 600), so the
9:55 bar (which closes at 10:00) is the 10:00 decision price.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Bands:
    upper: float
    lower: float
    sigma: float
    anchor_high: float
    anchor_low: float


def to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed in New York time."""
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC").tz_convert(ET)
    else:
        out.index = out.index.tz_convert(ET)
    return out.sort_index()


def session_date(ts: pd.Timestamp) -> date:
    ts = pd.Timestamp(ts)
    return (ts.tz_convert(ET) if ts.tzinfo else ts).date()


def slot_end(ts: pd.Timestamp, bar_minutes: int) -> int:
    """Minutes since midnight for the bar's end time."""
    end = pd.Timestamp(ts) + pd.Timedelta(minutes=bar_minutes)
    end = end.tz_convert(ET) if end.tzinfo else end
    return end.hour * 60 + end.minute


def group_sessions(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split a bar frame into regular-session days, sorted."""
    frame = to_et(df)
    if frame.empty:
        return {}
    frame = frame.between_time("09:30", "15:59")
    if frame.empty:
        return {}
    return {d: g.sort_index() for d, g in frame.groupby(frame.index.date)}


def session_open(frame: pd.DataFrame) -> float:
    return float(frame["open"].iloc[0])


def session_close(frame: pd.DataFrame) -> float:
    return float(frame["close"].iloc[-1])


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    """Cumulative regular-session VWAP using typical price (H+L+C)/3."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].astype(float).clip(lower=0.0)
    cumulative_volume = volume.cumsum()
    cumulative_pv = (typical * volume).cumsum()
    vwap = cumulative_pv / cumulative_volume.replace(0.0, np.nan)
    vwap = vwap.ffill().fillna(typical)
    return vwap


def sigma_by_slot(
    prior_sessions: list[pd.DataFrame],
    bar_minutes: int = 5,
    lookback: int = 14,
) -> dict[int, float]:
    """Average absolute percentage move from the open for every bar-end slot.

    Only ``prior_sessions`` are used, so the result is safe for today's open.
    """
    sessions = prior_sessions[-lookback:]
    if not sessions:
        return {}
    pieces: list[pd.DataFrame] = []
    for frame in sessions:
        if frame.empty:
            continue
        open_price = session_open(frame)
        if open_price <= 0:
            continue
        ends = frame.index + pd.Timedelta(minutes=bar_minutes)
        slots = np.asarray(ends.hour * 60 + ends.minute)
        moves = np.abs(frame["close"].to_numpy(float) / open_price - 1.0)
        pieces.append(pd.DataFrame({"slot": slots, "move": moves}))
    if not pieces:
        return {}
    combined = pd.concat(pieces, ignore_index=True)
    return {int(k): float(v) for k, v in combined.groupby("slot")["move"].mean().items()}


def sigma_at(sigma: dict[int, float], slot: int) -> float | None:
    """Sigma for a slot, falling back to the nearest earlier slot."""
    if not sigma:
        return None
    if slot in sigma:
        return sigma[slot]
    earlier = [s for s in sigma if s <= slot]
    if earlier:
        return sigma[max(earlier)]
    return sigma[min(sigma)]


def bands_for_slot(
    open_price: float,
    prior_close: float,
    sigma: dict[int, float],
    slot: int,
    mult: float = 1.0,
) -> Bands | None:
    """Bands anchored on today's open and yesterday's close (gap-aware)."""
    width = sigma_at(sigma, slot)
    if width is None or open_price <= 0 or prior_close <= 0:
        return None
    anchor_high = max(open_price, prior_close)
    anchor_low = min(open_price, prior_close)
    return Bands(
        upper=anchor_high * (1.0 + mult * width),
        lower=anchor_low * (1.0 - mult * width),
        sigma=width,
        anchor_high=anchor_high,
        anchor_low=anchor_low,
    )


def prepare_session(frame: pd.DataFrame, bar_minutes: int = 5) -> pd.DataFrame:
    """Add slot and VWAP columns to one session."""
    out = to_et(frame)
    ends = out.index + pd.Timedelta(minutes=bar_minutes)
    out["slot"] = np.asarray(ends.hour * 60 + ends.minute)
    out["vwap"] = session_vwap(out)
    return out


def is_decision_slot(slot: int, start: time, end: time, step_minutes: int = 30) -> bool:
    """True when a bar-end slot is one of the half-hourly decision points."""
    minutes = slot
    start_m = start.hour * 60 + start.minute
    end_m = end.hour * 60 + end.minute
    return start_m <= minutes <= end_m and (minutes - start_m) % step_minutes == 0


def next_decision_slot(slot: int, start: time, end: time, step_minutes: int = 30) -> int | None:
    """The first decision slot at or after ``slot``."""
    start_m = start.hour * 60 + start.minute
    end_m = end.hour * 60 + end.minute
    if slot > end_m:
        return None
    if slot <= start_m:
        return start_m
    remainder = (slot - start_m) % step_minutes
    candidate = slot if remainder == 0 else slot + (step_minutes - remainder)
    return candidate if candidate <= end_m else None


def flat_slot(s: time) -> int:
    return s.hour * 60 + s.minute


def slot_time(slot: int) -> time:
    return time(slot // 60, slot % 60)


def add_minutes(t: time, minutes: int) -> time:
    total = t.hour * 60 + t.minute + minutes
    return time(total // 60, total % 60)


def bar_end_timestamp(ts: pd.Timestamp, bar_minutes: int) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.Timedelta(minutes=bar_minutes)


def trading_days_between(start: date, end: date) -> int:
    """Calendar days, used only to size the data request."""
    return max((end - start).days, 1)
