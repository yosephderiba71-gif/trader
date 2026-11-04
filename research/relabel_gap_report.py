"""Relabel the gap report with the supplied NYSE calendar and list residual gaps."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from market_data import normalize_ohlcv
from research.check_data import _missing_hours, _weekend_gap_mask
from research.gap_report import holiday_windows, hour_holiday_labels

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data/XAUUSD_full_5m_clean.csv"
GAP_REPORT = ROOT / "research/gap_report.csv"
TARGET_HOURS = ROOT / "research/gap_report_91_hours.csv"
REMAINING_UNKNOWN = ROOT / "research/gap_report_remaining_unknown.csv"
INCOMPLETE_HOURS = ROOT / "research/incomplete_hours.csv"
SOURCE_NOTE = "User-supplied NYSE holiday list pasted in chat on 2026-11-04"

NYSE_HOLIDAYS: dict[int, tuple[str, ...]] = {
    2020: (
        "2020-01-01",
        "2020-01-20",
        "2020-02-17",
        "2020-04-10",
        "2020-05-25",
        "2020-07-03",
        "2020-09-07",
        "2020-11-26",
        "2020-12-25",
    ),
    2021: (
        "2021-01-01",
        "2021-01-18",
        "2021-02-15",
        "2021-04-02",
        "2021-05-31",
        "2021-07-05",
        "2021-09-06",
        "2021-11-25",
        "2021-12-24",
    ),
    2022: (
        "2022-01-17",
        "2022-02-21",
        "2022-04-15",
        "2022-05-30",
        "2022-06-20",
        "2022-07-04",
        "2022-09-05",
        "2022-11-24",
        "2022-12-26",
    ),
    2023: (
        "2023-01-02",
        "2023-01-16",
        "2023-02-20",
        "2023-04-07",
        "2023-05-29",
        "2023-06-19",
        "2023-07-04",
        "2023-09-04",
        "2023-11-23",
        "2023-12-25",
    ),
    2024: (
        "2024-01-01",
        "2024-01-15",
        "2024-02-19",
        "2024-03-29",
        "2024-05-27",
        "2024-06-19",
        "2024-07-04",
        "2024-09-02",
        "2024-11-28",
        "2024-12-25",
    ),
    2025: (
        "2025-01-01",
        "2025-01-20",
        "2025-02-17",
        "2025-04-18",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-12-25",
    ),
    2026: (
        "2026-01-01",
        "2026-01-19",
        "2026-02-16",
        "2026-04-03",
        "2026-05-25",
        "2026-06-19",
        "2026-07-03",
        "2026-09-07",
        "2026-11-26",
        "2026-12-25",
    ),
}


def supplied_dates() -> set[date]:
    return {
        date.fromisoformat(value)
        for values in NYSE_HOLIDAYS.values()
        for value in values
    }


def _write_iso_z(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in ("start", "end", "end_exclusive", "timestamp", "run_start", "run_end"):
        if column in output.columns and pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = (
                pd.to_datetime(output[column], utc=True)
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
    output.to_csv(path, index=False)


def derived_early_close_dates() -> set[date]:
    dates: set[date] = set()
    for year, values in NYSE_HOLIDAYS.items():
        holidays = {date.fromisoformat(value) for value in values}
        thanksgiving = next(
            day for day in holidays if day.month == 11 and day.weekday() == 3
        )
        dates.add(thanksgiving + timedelta(days=1))
        christmas_eve = date(year, 12, 24)
        if christmas_eve not in holidays:
            dates.add(christmas_eve)
    return dates


def main() -> None:
    supplied = supplied_dates()
    early_close_dates = derived_early_close_dates()

    raw = pd.read_csv(CLEAN_PATH)
    clean = normalize_ohlcv(raw, "XAUUSD", source=str(CLEAN_PATH))
    missing = _missing_hours(clean)
    weekend = _weekend_gap_mask(missing)
    non_weekend = missing[~weekend.to_numpy()]
    mon_thu = non_weekend[
        non_weekend.day_of_week.isin([0, 1, 2, 3]) & (non_weekend.hour < 21)
    ]

    old_windows = holiday_windows()
    old_holiday = pd.Series(
        [hour_holiday_labels(timestamp, old_windows) != "unknown" for timestamp in mon_thu],
        index=mon_thu,
    )
    target = mon_thu[~old_holiday.to_numpy()]
    if len(target) != 91:
        raise ValueError(f"expected 91 target hours, found {len(target)}")

    missing_dates = sorted({timestamp.date() for timestamp in target} - supplied)
    if missing_dates:
        raise ValueError(f"target dates not in supplied NYSE list: {missing_dates}")

    report = pd.read_csv(GAP_REPORT)
    report["start"] = pd.to_datetime(report["start"], utc=True)
    report["end"] = pd.to_datetime(report["end"], utc=True)
    report["end_exclusive"] = pd.to_datetime(report["end_exclusive"], utc=True)
    report["official_calendar_hours"] = 0
    report["official_calendar_dates"] = ""

    run_dates: dict[int, set[str]] = {}
    target_rows: list[dict[str, object]] = []
    for timestamp in target:
        matches = report.index[
            (report["start"] <= timestamp) & (report["end"] >= timestamp)
        ]
        if len(matches) != 1:
            raise ValueError(f"target hour {timestamp} matched {len(matches)} runs")
        index = int(matches[0])
        report.at[index, "official_calendar_hours"] += 1
        run_dates.setdefault(index, set()).add(str(timestamp.date()))
        target_rows.append(
            {
                "timestamp": timestamp,
                "date": str(timestamp.date()),
                "weekday": timestamp.day_name(),
                "run_start": report.at[index, "start"],
                "run_end": report.at[index, "end"],
            }
        )

    for index, dates in run_dates.items():
        report.at[index, "holiday"] = "Holiday/Early Close"
        report.at[index, "official_calendar_dates"] = ";".join(sorted(dates))

    extra_labeled: list[dict[str, object]] = []
    extra_candidates = report[
        report["holiday"].isin(["unknown", "Holiday/Early Close"])
        & (report["official_calendar_hours"] == 0)
    ]
    for index, row in extra_candidates.iterrows():
        run_hours = pd.date_range(row["start"], row["end"], freq="1h", tz="UTC")
        run_dates_set = {timestamp.date() for timestamp in run_hours}
        if run_dates_set and all(day in supplied for day in run_dates_set):
            report.at[index, "holiday"] = "Holiday/Early Close"
            report.at[index, "official_calendar_dates"] = ";".join(
                sorted(str(day) for day in run_dates_set)
            )
            extra_labeled.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "hours": int(row["hours"]),
                }
            )

    if int(report["official_calendar_hours"].sum()) != 91:
        raise ValueError("official_calendar_hours does not sum to 91")
    if int(report["hours"].sum()) != 2347:
        raise ValueError("run-hour total changed from 2347")
    if len(report) != 1771:
        raise ValueError("run count changed from 1771")

    _write_iso_z(report, GAP_REPORT)
    _write_iso_z(pd.DataFrame(target_rows), TARGET_HOURS)

    remaining = report[report["holiday"] == "unknown"].copy()
    _write_iso_z(remaining, REMAINING_UNKNOWN)
    daily_break = remaining[
        (remaining["hours"] == 1) & remaining["start"].dt.hour.isin([21, 22])
    ]
    non_break = remaining.drop(daily_break.index)

    counts = clean.groupby(clean.index.floor("1h")).size()
    incomplete = counts[(counts > 0) & (counts != 12)]
    incomplete_rows: list[dict[str, object]] = []
    for timestamp, bars in incomplete.items():
        day = timestamp.date()
        is_early_close_date = day in early_close_dates
        is_early_close_hour = bool(
            is_early_close_date and int(bars) == 9 and timestamp.hour in (17, 18, 19)
        )
        if day in supplied:
            calendar_status = "full holiday (supplied)"
        elif is_early_close_date:
            calendar_status = "early-close date (derived)"
        else:
            calendar_status = "regular day"
        incomplete_rows.append(
            {
                "timestamp": timestamp,
                "bars": int(bars),
                "date": str(day),
                "weekday": timestamp.day_name(),
                "calendar_status": calendar_status,
                "early_close_date": "yes" if is_early_close_date else "no",
                "early_close": "yes" if is_early_close_hour else "no",
            }
        )
    incomplete_frame = pd.DataFrame(incomplete_rows)
    _write_iso_z(incomplete_frame, INCOMPLETE_HOURS)

    print(f"source_note: {SOURCE_NOTE}")
    print(f"supplied_holiday_dates: {len(supplied)}")
    print(f"derived_early_close_dates: {len(early_close_dates)}")
    print(f"target_hours: {len(target)}")
    print(f"target_runs: {len(run_dates)}")
    print(f"extra_calendar_runs: {len(extra_labeled)}")
    for row in extra_labeled:
        print(
            f"  extra: {row['start']} -> {row['end']} hours={row['hours']}"
        )
    print("holiday_counts:")
    print(report["holiday"].value_counts().to_string())
    print(f"official_calendar_hours_sum: {int(report['official_calendar_hours'].sum())}")
    print(f"run_hours_sum: {int(report['hours'].sum())}")
    print(f"runs_total: {len(report)}")
    print(f"remaining_unknown_runs: {len(remaining)}")
    print(f"remaining_unknown_daily_break: {len(daily_break)}")
    print(f"remaining_unknown_non_break: {len(non_break)}")
    print("remaining_unknown_non_break_runs:")
    for row in non_break.sort_values("start").itertuples(index=False):
        print(
            f"  {row.start} -> {row.end} hours={int(row.hours)} "
            f"holiday={row.holiday}"
        )
    print(f"incomplete_hours: {len(incomplete_frame)}")
    print("incomplete_calendar_status:")
    print(incomplete_frame["calendar_status"].value_counts().to_string())
    print("incomplete_early_close:")
    print(incomplete_frame["early_close"].value_counts().to_string())
    print(f"target_hours_csv: {TARGET_HOURS}")
    print(f"remaining_unknown_csv: {REMAINING_UNKNOWN}")
    print(f"incomplete_hours_csv: {INCOMPLETE_HOURS}")


if __name__ == "__main__":
    main()
