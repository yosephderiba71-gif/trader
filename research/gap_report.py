from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from dateutil.easter import easter

ROOT = Path("/Users/yosi/Desktop/trading-agent")
sys.path.insert(0, str(ROOT))

from market_data import normalize_ohlcv
from research.check_data import _missing_hours, _weekend_gap_mask

CLEAN_PATH = ROOT / "data/XAUUSD_full_5m_clean.csv"
GAP_REPORT = ROOT / "research/gap_report.csv"
YEARS = range(2020, 2027)


def holiday_windows() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for year in YEARS:
        windows.append(
            (
                "Christmas",
                pd.Timestamp(f"{year}-12-24 18:00", tz="UTC"),
                pd.Timestamp(f"{year}-12-26 23:59:59", tz="UTC"),
            )
        )
        windows.append(
            (
                "New Year",
                pd.Timestamp(f"{year}-12-31 18:00", tz="UTC"),
                pd.Timestamp(f"{year + 1}-01-02 23:59:59", tz="UTC"),
            )
        )
        easter_sunday = easter(year)
        good_friday = easter_sunday - timedelta(days=2)
        easter_monday = easter_sunday + timedelta(days=1)
        for name, day in (("Good Friday", good_friday), ("Easter Monday", easter_monday)):
            windows.append(
                (
                    name,
                    pd.Timestamp(day, tz="UTC"),
                    pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                )
            )
    return windows


def holiday_labels(start: pd.Timestamp, end_exclusive: pd.Timestamp, windows) -> str:
    labels: list[str] = []
    for name, window_start, window_end in windows:
        if start < window_end and end_exclusive > window_start:
            if name not in labels:
                labels.append(name)
    return "; ".join(labels) if labels else "unknown"


def hour_holiday_labels(timestamp: pd.Timestamp, windows) -> str:
    labels: list[str] = []
    for name, window_start, window_end in windows:
        if window_start <= timestamp <= window_end:
            if name not in labels:
                labels.append(name)
    return "; ".join(labels) if labels else "unknown"


def main() -> None:
    raw = pd.read_csv(CLEAN_PATH)
    clean = normalize_ohlcv(raw, "XAUUSD", source=str(CLEAN_PATH))
    missing = _missing_hours(clean)
    weekend = _weekend_gap_mask(missing)
    non_weekend = missing[~weekend.to_numpy()]

    runs: list[dict[str, object]] = []
    if len(non_weekend):
        start = previous = non_weekend[0]
        hours = 1
        for timestamp in non_weekend[1:]:
            if timestamp - previous == pd.Timedelta(hours=1):
                hours += 1
            else:
                runs.append({"start": start, "end": previous, "hours": hours})
                start = timestamp
                hours = 1
            previous = timestamp
        runs.append({"start": start, "end": previous, "hours": hours})

    windows = holiday_windows()
    for run in runs:
        start = run["start"]
        end = run["end"]
        assert isinstance(start, pd.Timestamp)
        assert isinstance(end, pd.Timestamp)
        run["end_exclusive"] = end + pd.Timedelta(hours=1)
        run["holiday"] = holiday_labels(start, run["end_exclusive"], windows)

    runs_df = pd.DataFrame(runs)
    runs_df["start"] = pd.to_datetime(runs_df["start"], utc=True)
    runs_df["end"] = pd.to_datetime(runs_df["end"], utc=True)
    runs_df["end_exclusive"] = pd.to_datetime(runs_df["end_exclusive"], utc=True)

    output = runs_df[["start", "end", "hours", "holiday", "end_exclusive"]].copy()
    for column in ("start", "end", "end_exclusive"):
        output[column] = output[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output.to_csv(GAP_REPORT, index=False)

    hourly = pd.DataFrame({"timestamp": non_weekend})
    hourly["year"] = hourly["timestamp"].dt.year
    hourly["weekday"] = hourly["timestamp"].dt.day_of_week
    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["holiday"] = hourly["timestamp"].apply(
        lambda timestamp: hour_holiday_labels(timestamp, windows)
    )
    hourly["is_holiday"] = hourly["holiday"] != "unknown"

    total_by_year = hourly.groupby("year").size().reindex(YEARS, fill_value=0)
    non_holiday_by_year = (
        hourly.loc[~hourly["is_holiday"]].groupby("year").size().reindex(YEARS, fill_value=0)
    )
    mon_thu = hourly[
        hourly["weekday"].isin([0, 1, 2, 3])
        & (hourly["hour"] < 21)
        & ~hourly["is_holiday"]
    ]
    mon_thu_inclusive = hourly[
        hourly["weekday"].isin([0, 1, 2, 3])
        & (hourly["hour"] <= 21)
        & ~hourly["is_holiday"]
    ]
    non_holiday_runs = runs_df[runs_df["holiday"] == "unknown"]
    runs_ge6_non_holiday = non_holiday_runs[non_holiday_runs["hours"] >= 6]

    print(f"clean_rows: {len(clean)}")
    print(f"missing_hours: {len(missing)}")
    print(f"missing_hours_weekend: {int(weekend.sum())}")
    print(f"missing_hours_non_weekend: {len(non_weekend)}")
    print(f"runs_total: {len(runs_df)}")
    print(f"runs_holiday: {int((runs_df['holiday'] != 'unknown').sum())}")
    print(f"runs_unknown: {int((runs_df['holiday'] == 'unknown').sum())}")
    print(f"runs_ge6_non_holiday: {len(runs_ge6_non_holiday)}")
    print(f"missing_hours_mon_thu_00_21_non_holiday: {len(mon_thu)}")
    print(f"missing_hours_mon_thu_00_21_inclusive_non_holiday: {len(mon_thu_inclusive)}")
    print("total_missing_non_weekend_hours_by_year:")
    for year, count in total_by_year.items():
        print(f"  {int(year)}: {int(count)}")
    print("non_holiday_missing_non_weekend_hours_by_year:")
    for year, count in non_holiday_by_year.items():
        print(f"  {int(year)}: {int(count)}")
    print(f"gap_report_csv: {GAP_REPORT}")
    print(f"gap_report_rows: {len(output)}")

    print("largest_non_holiday_runs:")
    largest = non_holiday_runs.sort_values(["hours", "start"], ascending=[False, True]).head(20)
    for row in largest.itertuples(index=False):
        print(
            f"  start={row.start} end={row.end} hours={int(row.hours)} holiday={row.holiday}"
        )


if __name__ == "__main__":
    main()
