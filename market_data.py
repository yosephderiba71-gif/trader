"""Read-only market data providers.

The strategy layer should depend only on ``MarketDataProvider``. It should not
import Alpaca or a CSV reader directly. Switching between the Alpaca paper feed
and a local CSV is then a constructor change, not a strategy change.

Nothing in this module can place, cancel, or close an order.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd

OHLCV = ("open", "high", "low", "close", "volume")
_TIMESTAMP_COLUMNS = ("timestamp", "datetime", "date", "time", "ts")
_ALIASES = {
    "xauusd_5m": "XAUUSD",
    "gold_daily_gc_f_2010": "GC=F",
    "spy_5m_yahoo": "SPY",
}

TimestampLike = datetime | pd.Timestamp | str | None


class MarketDataProvider(Protocol):
    """Minimal read-only contract shared by every data source."""

    name: str

    def list_symbols(self) -> list[str]:
        ...

    def get_bars(
        self,
        symbol: str,
        start: TimestampLike = None,
        end: TimestampLike = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        ...


def _as_utc(value: TimestampLike) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _empty_bars(symbol: str, source: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC"))
    frame.attrs.update(symbol=symbol, source=str(source), invalid_ohlc=0)
    return frame


def _coerce_index(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df.index = index.rename(None)
        return df

    lower = {str(column).strip().lower(): column for column in df.columns}
    timestamp_column = next((lower[name] for name in _TIMESTAMP_COLUMNS if name in lower), None)
    if timestamp_column is None:
        first = df.columns[0]
        if str(first).startswith("Unnamed"):
            timestamp_column = first
        else:
            raise ValueError("no timestamp/date column found")

    values = df[timestamp_column]
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        sample = numeric.dropna()
        unit = "ms" if len(sample) and abs(float(sample.iloc[0])) > 10_000_000_000 else "s"
        index = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        index = pd.to_datetime(values, utc=True, errors="coerce")

    df = df.drop(columns=[timestamp_column])
    df.index = pd.DatetimeIndex(index).rename(None)
    return df


def normalize_ohlcv(frame: pd.DataFrame, symbol: str, source: str = "") -> pd.DataFrame:
    """Normalize any provider frame to UTC DatetimeIndex + OHLCV columns.

    The returned frame is sorted, de-duplicated, and has a ``volume`` column
    (zero when the source has no volume). ``frame.attrs`` records the symbol,
    source, and the number of OHLC envelope violations rather than silently
    repairing them.
    """
    if frame is None or frame.empty:
        return _empty_bars(symbol, source)

    df = _coerce_index(frame)
    df.columns = [str(column).strip().lower() for column in df.columns]

    missing = [column for column in ("open", "high", "low", "close") if column not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing required columns: {', '.join(missing)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[list(OHLCV)]
    for column in OHLCV:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[~df.index.isna()]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    invalid = int(
        (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        ).sum()
    )
    df.attrs.update(symbol=symbol, source=str(source), invalid_ohlc=invalid)
    return df


def infer_symbol(stem: str) -> str:
    key = stem.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    for suffix in ("_5m", "_1m", "_15m", "_1h", "_daily", "_yahoo"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key.upper()


def discover_csv_files(roots: Sequence[str | Path] = ("data", "intraday")) -> dict[str, Path]:
    """Find CSVs with OHLC columns and at least one valid bar."""
    found: dict[str, Path] = {}
    for root in roots:
        directory = Path(root)
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
                normalized = normalize_ohlcv(frame, infer_symbol(path.stem), source=str(path))
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            if not normalized.empty:
                found[infer_symbol(path.stem)] = path
    return found


class CsvMarketDataProvider:
    """Read-only local CSV provider for deterministic backtests."""

    name = "csv"

    def __init__(
        self,
        files: Mapping[str, str | Path] | None = None,
        roots: Sequence[str | Path] = ("data", "intraday"),
    ):
        self._files = (
            {str(symbol): Path(path) for symbol, path in files.items()}
            if files is not None
            else discover_csv_files(roots)
        )

    @property
    def files(self) -> dict[str, Path]:
        return dict(self._files)

    def list_symbols(self) -> list[str]:
        return sorted(self._files)

    def get_bars(
        self,
        symbol: str,
        start: TimestampLike = None,
        end: TimestampLike = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        path = self._files.get(symbol)
        if path is None:
            raise KeyError(f"unknown symbol: {symbol}")

        raw = pd.read_csv(path)
        frame = normalize_ohlcv(raw, symbol, source=str(path))
        start_ts = _as_utc(start)
        end_ts = _as_utc(end)
        if start_ts is not None:
            frame = frame[frame.index >= start_ts]
        if end_ts is not None:
            frame = frame[frame.index <= end_ts]
        frame.attrs.update(
            symbol=symbol,
            source=str(path),
            invalid_ohlc=frame.attrs.get("invalid_ohlc", 0),
        )
        return frame


def _alpaca_timeframe(name: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    mapping = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if name not in mapping:
        raise ValueError(f"unsupported timeframe: {name}")
    return mapping[name]


class AlpacaMarketDataProvider:
    """Read-only Alpaca bars provider.

    It uses only market-data endpoints and has no order methods.
    """

    name = "alpaca"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: Sequence[str],
        feed: str = "iex",
    ):
        if not api_key or not api_secret:
            raise ValueError("Alpaca data provider requires paper API credentials")
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        self._feed = feed

    def list_symbols(self) -> list[str]:
        return list(self._symbols)

    def get_bars(
        self,
        symbol: str,
        start: TimestampLike = None,
        end: TimestampLike = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest

        start_ts = _as_utc(start) or (datetime.now(timezone.utc) - timedelta(days=180))
        end_ts = _as_utc(end) or datetime.now(timezone.utc)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_alpaca_timeframe(timeframe),
            start=start_ts.to_pydatetime(),
            end=end_ts.to_pydatetime(),
            feed=DataFeed(self._feed),
        )
        raw = self._client.get_stock_bars(request).df
        if raw.empty:
            return _empty_bars(symbol, f"alpaca:{self._feed}")
        if isinstance(raw.index, pd.MultiIndex):
            if symbol not in raw.index.get_level_values(0):
                return _empty_bars(symbol, f"alpaca:{self._feed}")
            raw = raw.xs(symbol, level=0)
        return normalize_ohlcv(raw, symbol, source=f"alpaca:{self._feed}")


def make_provider(kind: str, **kwargs) -> MarketDataProvider:
    if kind == "csv":
        return CsvMarketDataProvider(**kwargs)
    if kind == "alpaca":
        return AlpacaMarketDataProvider(**kwargs)
    raise ValueError(f"unknown provider: {kind}")
