from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "check_data.py"


def _make_csv(path: Path) -> None:
    index = pd.date_range("2024-01-01", "2024-01-02 23:55", freq="5min", tz="UTC")
    close = 100.0 + pd.Series(range(len(index)), index=index) * 0.1
    frame = pd.DataFrame(
        {
            "timestamp": index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": close.values - 0.2,
            "high": close.values + 1.0,
            "low": close.values - 1.0,
            "close": close.values,
        }
    )
    frame.loc[frame.index[0], ["open", "high", "low", "close"]] = 100.0
    frame = frame[frame["timestamp"] != "2024-01-01T05:05:00Z"]

    duplicate = frame.loc[frame["timestamp"] == "2024-01-01T06:00:00Z"].iloc[0].copy()
    duplicate["close"] = duplicate["close"] + 0.5
    frame = pd.concat([frame, duplicate.to_frame().T], ignore_index=True)
    frame.to_csv(path, index=False)


def _make_market_hours_csv(
    path: Path,
    start: str,
    end: str,
    flat_at: str,
) -> None:
    index = pd.date_range(start, end, freq="5min", tz="UTC")
    weekend_start = pd.Timestamp("2024-01-05 22:00", tz="UTC")
    weekend_end = pd.Timestamp("2024-01-07 22:00", tz="UTC")
    index = index[(index < weekend_start) | (index >= weekend_end)]

    close = 100.0 + pd.Series(range(len(index)), index=index) * 0.1
    frame = pd.DataFrame(
        {
            "timestamp": index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": close.values - 0.2,
            "high": close.values + 1.0,
            "low": close.values - 1.0,
            "close": close.values,
        }
    )
    frame.loc[frame["timestamp"] == flat_at, ["open", "high", "low", "close"]] = 100.0
    frame.to_csv(path, index=False)


def _run(
    paths: Path | list[Path],
    start: str,
    end: str,
) -> subprocess.CompletedProcess[str]:
    if isinstance(paths, Path):
        paths = [paths]
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(path) for path in paths), start, end],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _parse(stdout: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            result[key] = value
    return result


def test_check_data_reports_synthetic_csv(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    _make_csv(path)

    completed = _run(path, "2024-01-01", "2024-01-02")
    assert completed.returncode == 0
    output = _parse(completed.stdout)

    assert output["source_count"] == "1"
    assert output["first_timestamp"] == "2024-01-01 00:00:00+00:00"
    assert output["last_timestamp"] == "2024-01-02 23:55:00+00:00"
    assert output["total_rows"] == "576"
    assert output["merged_rows"] == "575"
    assert output["flat_bars"] == "1"
    assert output["missing_hours"] == "0"
    assert output["missing_hours_weekend"] == "0"
    assert output["missing_hours_other"] == "0"
    assert output["incomplete_hours"] == "1"
    assert output["missing_or_incomplete_hours"] == "1"
    assert output["duplicate_timestamps"] == "1"
    assert output["all_timestamps_utc"] == "True"
    assert output["all_on_5m_grid"] == "True"
    assert output["first_timestamp_is_2020_01_01"] == "False"
    assert output["last_timestamp_in_2026"] == "False"
    assert output["date_range_ok"] == "True"


def test_check_data_merges_sources_and_classifies_gaps(tmp_path: Path) -> None:
    source1 = tmp_path / "source1.csv"
    source2 = tmp_path / "source2.csv"
    _make_csv(source1)
    _make_market_hours_csv(
        source2,
        "2024-01-05",
        "2024-01-08 23:55",
        "2024-01-05T00:00:00Z",
    )

    completed = _run([source1, source2], "2024-01-01", "2024-01-08")
    assert completed.returncode == 0
    output = _parse(completed.stdout)

    assert output["source_count"] == "2"
    assert output["total_rows"] == "1152"
    assert output["merged_rows"] == "1151"
    assert output["flat_bars"] == "2"
    assert output["flat_bars_before_merge_total"] == "2"
    assert output["source_1_flat_bars_before_merge"] == "1"
    assert output["source_2_flat_bars_before_merge"] == "1"
    assert output["source_1_flat_bars_after_merge"] == "1"
    assert output["source_2_flat_bars_after_merge"] == "1"
    assert output["missing_hours"] == "96"
    assert output["missing_hours_weekend"] == "48"
    assert output["missing_hours_other"] == "48"
    assert output["incomplete_hours"] == "1"
    assert output["missing_or_incomplete_hours"] == "97"
    assert output["duplicate_timestamps"] == "1"
    assert output["first_timestamp"] == "2024-01-01 00:00:00+00:00"
    assert output["last_timestamp"] == "2024-01-08 23:55:00+00:00"
    assert output["first_timestamp_is_2020_01_01"] == "False"
    assert output["last_timestamp_in_2026"] == "False"
    assert "not in 2026" in output["note"]


def test_check_data_fails_on_date_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    _make_csv(path)

    completed = _run(path, "2024-01-01", "2024-01-03")
    assert completed.returncode != 0
    output = _parse(completed.stdout)
    assert output["date_range_ok"] == "False"
