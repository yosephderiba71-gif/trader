"""Check one or more five-minute OHLCV CSVs, merge them, and report gaps."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from market_data import normalize_ohlcv

_TIMESTAMP_COLUMNS = ("timestamp", "datetime", "date", "time", "ts")
_BARS_PER_HOUR = 12
_WEEKEND_CLOSE_HOUR = 22
_WEEKEND_OPEN_HOUR = 22


def _timestamp_column(raw: pd.DataFrame):
    lower = {str(column).strip().lower(): column for column in raw.columns}
    column = next((lower[name] for name in _TIMESTAMP_COLUMNS if name in lower), None)
    if column is None:
        column = raw.columns[0]
    return column


def _timestamp_series(raw: pd.DataFrame) -> pd.Series:
    values = raw[_timestamp_column(raw)]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        sample = numeric.dropna()
        unit = "ms" if len(sample) and abs(float(sample.iloc[0])) > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _raw_timestamps_utc(raw: pd.DataFrame) -> bool:
    values = raw[_timestamp_column(raw)]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return True

    try:
        parsed = pd.to_datetime(values, utc=False, errors="coerce")
    except (TypeError, ValueError):
        return False

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return str(parsed.dt.tz) == "UTC"
    if parsed.dtype == object:
        valid = parsed.dropna()
        if len(valid) != len(parsed):
            return False
        for value in valid:
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError):
                return False
            if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                return False
        return True
    return False


def _flat_mask(bars: pd.DataFrame) -> pd.Series:
    return (
        (bars["open"] == bars["high"])
        & (bars["high"] == bars["low"])
        & (bars["low"] == bars["close"])
    )


def _flat_count(bars: pd.DataFrame) -> int:
    return int(_flat_mask(bars).sum())


def _missing_hours(
    bars: pd.DataFrame,
    bars_per_hour: int = _BARS_PER_HOUR,
) -> pd.DatetimeIndex:
    if bars.empty:
        return pd.DatetimeIndex([], tz="UTC")

    counts = bars.groupby(bars.index.floor("1h")).size()
    full_range = pd.date_range(
        bars.index.min().floor("1h"),
        bars.index.max().floor("1h"),
        freq="1h",
        tz="UTC",
    )
    return full_range.difference(counts.index)


def _incomplete_hours(
    bars: pd.DataFrame,
    bars_per_hour: int = _BARS_PER_HOUR,
) -> int:
    if bars.empty:
        return 0
    counts = bars.groupby(bars.index.floor("1h")).size()
    return int(((counts > 0) & (counts != bars_per_hour)).sum())


def _weekend_gap_mask(hours: pd.DatetimeIndex) -> pd.Series:
    weekday = hours.day_of_week
    hour = hours.hour
    mask = (
        (weekday == 5)
        | ((weekday == 6) & (hour < _WEEKEND_OPEN_HOUR))
        | ((weekday == 4) & (hour >= _WEEKEND_CLOSE_HOUR))
    )
    return pd.Series(mask, index=hours)


def _utc_and_grid(timestamps: pd.DatetimeIndex) -> tuple[bool, bool]:
    valid = timestamps[~timestamps.isna()]
    if len(valid) == 0:
        return False, False

    all_utc = str(valid.tz) == "UTC"
    all_on_grid = bool(
        ((valid.minute % 5) == 0).all()
        and (valid.second == 0).all()
        and (valid.microsecond == 0).all()
    )
    return all_utc, all_on_grid


def check_data(
    csv_paths: Sequence[Path],
    expected_start: date,
    expected_end: date,
    strict: bool = False,
) -> dict[str, object]:
    if not csv_paths:
        raise ValueError("at least one CSV path is required")

    sources: list[dict[str, object]] = []
    timestamp_parts: list[pd.DatetimeIndex] = []

    for path in csv_paths:
        raw = pd.read_csv(path)
        if raw.empty:
            raise ValueError(f"{path} is empty")

        timestamps = pd.DatetimeIndex(_timestamp_series(raw))
        bars = normalize_ohlcv(raw, "DATA", source=str(path))
        if bars.empty:
            raise ValueError(f"{path} has no valid OHLCV bars")

        sources.append(
            {
                "path": path,
                "raw_rows": int(len(raw)),
                "raw_utc": _raw_timestamps_utc(raw),
                "bars": bars,
                "flat_before": _flat_count(bars),
            }
        )
        timestamp_parts.append(timestamps)

    all_timestamps = pd.DatetimeIndex(
        pd.concat([pd.Series(part) for part in timestamp_parts], ignore_index=True)
    )
    valid_timestamps = all_timestamps[~all_timestamps.isna()]
    duplicate_timestamps = int(valid_timestamps.duplicated().sum())
    all_utc, all_on_grid = _utc_and_grid(all_timestamps)

    tagged: list[pd.DataFrame] = []
    for source_index, source in enumerate(sources, start=1):
        bars = source["bars"]
        tagged_bars = bars.copy()
        tagged_bars["_source_index"] = source_index
        tagged.append(tagged_bars)

    merged = pd.concat(tagged, axis=0)
    merged = merged.sort_index(kind="stable")
    merged = merged[~merged.index.duplicated(keep="last")]

    merged_rows = int(len(merged))
    merged_flat = _flat_count(merged)
    flat_mask = _flat_mask(merged)
    after_flat = flat_mask.groupby(merged["_source_index"]).sum()
    after_rows = merged.groupby("_source_index").size()

    missing = _missing_hours(merged)
    weekend_mask = _weekend_gap_mask(missing)
    missing_weekend = int(weekend_mask.sum())
    missing_other = int(len(missing) - missing_weekend)
    incomplete = _incomplete_hours(merged)

    first = merged.index.min()
    last = merged.index.max()
    first_is_2020 = first.date() == date(2020, 1, 1)
    last_in_2026 = last.year == 2026
    date_range_ok = first.date() == expected_start and last.date() == expected_end

    if strict:
        problems: list[str] = []
        for source in sources:
            if not bool(source["raw_utc"]):
                problems.append(f"{source['path']} has non-UTC or naive timestamps")
        if len(valid_timestamps) != len(all_timestamps):
            problems.append("some timestamps are invalid or unparseable")
        if not all_utc:
            problems.append("timestamps are not UTC")
        if not all_on_grid:
            problems.append("timestamps are not exactly on the 5-minute grid")
        if not date_range_ok:
            problems.append(
                f"first/last date {first.date()}..{last.date()} "
                f"does not match expected {expected_start}..{expected_end}"
            )
        if problems:
            raise ValueError("strict validation failed: " + "; ".join(problems))

    result: dict[str, object] = {
        "source_count": len(sources),
        "total_rows": int(sum(int(source["raw_rows"]) for source in sources)),
        "merged_rows": merged_rows,
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "first_timestamp_is_2020_01_01": first_is_2020,
        "last_timestamp_in_2026": last_in_2026,
        "flat_bars": merged_flat,
        "flat_bars_before_merge_total": int(
            sum(int(source["flat_before"]) for source in sources)
        ),
        "missing_hours": len(missing),
        "missing_hours_weekend": missing_weekend,
        "missing_hours_other": missing_other,
        "weekend_gap_window": "Friday 22:00 UTC to Sunday 22:00 UTC",
        "incomplete_hours": incomplete,
        "missing_or_incomplete_hours": len(missing) + incomplete,
        "duplicate_timestamps": duplicate_timestamps,
        "all_timestamps_utc": all_utc,
        "all_on_5m_grid": all_on_grid,
        "expected_start": str(expected_start),
        "expected_end": str(expected_end),
        "date_range_ok": date_range_ok,
        "strict": strict,
    }

    for source_index, source in enumerate(sources, start=1):
        result[f"source_{source_index}"] = str(source["path"])
        result[f"source_{source_index}_raw_rows"] = int(source["raw_rows"])
        result[f"source_{source_index}_flat_bars_before_merge"] = int(
            source["flat_before"]
        )
        result[f"source_{source_index}_rows_after_merge"] = int(
            after_rows.get(source_index, 0)
        )
        result[f"source_{source_index}_flat_bars_after_merge"] = int(
            after_flat.get(source_index, 0)
        )

    notes: list[str] = []
    if not first_is_2020:
        notes.append(f"first timestamp is {first}, not 2020-01-01")
    if not last_in_2026:
        notes.append(f"last timestamp is {last}, not in 2026; no gap fill performed")
    if notes:
        result["note"] = "; ".join(notes)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("expected_start", type=date.fromisoformat)
    parser.add_argument("expected_end", type=date.fromisoformat)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="reject non-UTC, off-grid, or unexpected-date timestamps",
    )
    args = parser.parse_args()

    try:
        result = check_data(
            args.csv_paths,
            args.expected_start,
            args.expected_end,
            strict=args.strict,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if bool(result["date_range_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
