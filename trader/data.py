"""Market data from Alpaca. Alpaca imports are local so the core logic stays importable without the SDK."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import Settings

OHLCV = ["open", "high", "low", "close", "volume"]


def drop_forming_bar(df: pd.DataFrame, now: datetime, timeframe_hours: int) -> pd.DataFrame:
    """Remove the bar that is still forming. Bar timestamps mark the bar's start."""
    if df.empty:
        return df
    if df.index[-1] + pd.Timedelta(hours=timeframe_hours) > pd.Timestamp(now):
        return df.iloc[:-1]
    return df


def fetch_bars(client, symbols, s: Settings, now: datetime | None = None) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    now = now or datetime.now(timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame(s.timeframe_hours, TimeFrameUnit.Hour),
        start=now - timedelta(days=s.lookback_days),
        end=now,
        feed=DataFeed(s.feed),
    )
    raw = client.get_stock_bars(request).df
    out: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out
    available = set(raw.index.get_level_values(0))
    for symbol in symbols:
        if symbol not in available:
            continue
        df = raw.xs(symbol, level=0)[OHLCV].sort_index()
        out[symbol] = drop_forming_bar(df, now, s.timeframe_hours)
    return out
