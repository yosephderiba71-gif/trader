"""Merge the three XAUUSD five-minute sources and verify the existing full file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from market_data import OHLCV, normalize_ohlcv
from research.check_data import (
    _raw_timestamps_utc,
    _timestamp_series,
    _utc_and_grid,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "data/XAUUSD_5m.csv",
    ROOT / "data/xauusd_2020_2022.csv",
    ROOT / "data/xauusd_2025_2026.csv",
)
OUTPUT = ROOT / "data/XAUUSD_full_5m.csv"
REPORT = ROOT / "research/merge_xauusd_full_report.json"
CANDIDATE = ROOT / "research/XAUUSD_full_5m_candidate.csv"


def _basis_hint(path: Path, raw: pd.DataFrame) -> str:
    tokens = [str(path).lower(), *(str(column).lower() for column in raw.columns)]
    has_bid = any("bid" in token for token in tokens)
    has_ask = any("ask" in token for token in tokens)
    if has_bid and has_ask:
        return "bid+ask"
    if has_bid:
        return "bid"
    if has_ask:
        return "ask"
    return "unspecified"


def _iso_z(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.reset_index()
    output = output.rename(columns={output.columns[0]: "timestamp"})
    output["timestamp"] = (
        pd.to_datetime(output["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return output[["timestamp", "open", "high", "low", "close", "volume"]]


def _compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    same_index = left.index.equals(right.index)
    same_columns = list(left.columns) == list(right.columns)
    result: dict[str, object] = {
        "same_index": same_index,
        "same_columns": same_columns,
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
    }
    if not same_index or not same_columns:
        result["identical"] = False
        return result

    close_diff = (left["close"] - right["close"]).abs()
    ohlc_diff = (
        (left[["open", "high", "low", "close"]] - right[["open", "high", "low", "close"]])
        .abs()
        .max(axis=1)
    )
    volume_diff = (left["volume"] - right["volume"]).abs()
    result.update(
        {
            "identical": bool(
                ohlc_diff.max() == 0 and volume_diff.max() == 0
            ),
            "max_ohlc_difference": float(ohlc_diff.max()),
            "max_close_difference": float(close_diff.max()),
            "max_volume_difference": float(volume_diff.max()),
            "different_ohlc_rows": int((ohlc_diff > 0).sum()),
            "different_volume_rows": int((volume_diff > 0).sum()),
        }
    )
    return result


def main() -> None:
    frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    source_names: list[str] = []

    for path in SOURCES:
        raw = pd.read_csv(path)
        raw_timestamps = pd.DatetimeIndex(_timestamp_series(raw))
        raw_utc = _raw_timestamps_utc(raw)
        all_utc, all_on_grid = _utc_and_grid(raw_timestamps)
        if not raw_utc:
            raise ValueError(f"{path} has non-UTC or naive timestamps")
        if not all_utc:
            raise ValueError(f"{path} timestamps are not UTC after parsing")
        if not all_on_grid:
            raise ValueError(f"{path} timestamps are not exactly on the 5-minute grid")
        if raw_timestamps.isna().any():
            raise ValueError(f"{path} has invalid or unparseable timestamps")

        bars = normalize_ohlcv(raw, "XAUUSD", source=str(path))
        duplicates_within = int(raw_timestamps.duplicated().sum())
        frames.append(bars)
        source_names.append(str(path.relative_to(ROOT)))
        source_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "raw_rows": int(len(raw)),
                "normalized_rows": int(len(bars)),
                "duplicates_within_source": duplicates_within,
                "first_timestamp": str(bars.index.min()),
                "last_timestamp": str(bars.index.max()),
                "first_close": float(bars["close"].iloc[0]),
                "last_close": float(bars["close"].iloc[-1]),
                "price_basis_hint": _basis_hint(path, raw),
            }
        )

    overlap_rows: list[dict[str, object]] = []
    for left_index, left in enumerate(frames):
        for right_index in range(left_index + 1, len(frames)):
            right = frames[right_index]
            common = left.index.intersection(right.index)
            row: dict[str, object] = {
                "left": source_names[left_index],
                "right": source_names[right_index],
                "overlap_rows": int(len(common)),
            }
            if len(common):
                difference = (left.loc[common, "close"] - right.loc[common, "close"]).abs()
                row.update(
                    {
                        "max_close_difference": float(difference.max()),
                        "mean_close_difference": float(difference.mean()),
                        "different_close_rows": int((difference > 0).sum()),
                    }
                )
            overlap_rows.append(row)

    basis_hints = {row["price_basis_hint"] for row in source_rows}
    overlap_count = sum(int(row["overlap_rows"]) for row in overlap_rows)
    overlap_differences = [
        float(row["max_close_difference"])
        for row in overlap_rows
        if "max_close_difference" in row
    ]
    if "bid" in basis_hints and "ask" in basis_hints:
        basis_status = "FLAGGED: sources declare different bid/ask basis"
        basis_flagged = True
    elif overlap_count and any(value > 0 for value in overlap_differences):
        basis_status = "FLAGGED: overlapping closes differ"
        basis_flagged = True
    elif overlap_count:
        basis_status = "same basis likely: overlapping closes match"
        basis_flagged = False
    else:
        basis_status = "UNVERIFIED: no bid/ask metadata and no overlapping timestamps"
        basis_flagged = True

    tagged = []
    for source_index, bars in enumerate(frames):
        tagged_bars = bars.copy()
        tagged_bars["_source_index"] = source_index
        tagged.append(tagged_bars)
    merged = pd.concat(tagged, axis=0).sort_index(kind="stable")
    duplicate_mask = merged.index.duplicated(keep="first")
    dropped_duplicate_rows = int(duplicate_mask.sum())
    merged = merged[~duplicate_mask]
    merged = merged.sort_index()

    existing = normalize_ohlcv(
        pd.read_csv(OUTPUT),
        "XAUUSD",
        source=str(OUTPUT),
    )
    comparison = _compare_frames(merged[list(OHLCV)], existing)

    first = merged.index.min()
    last = merged.index.max()
    report: dict[str, object] = {
        "sources": source_rows,
        "overlaps": overlap_rows,
        "overlap_rows_total": overlap_count,
        "price_basis_hints": sorted(basis_hints),
        "price_basis_status": basis_status,
        "price_basis_flagged": basis_flagged,
        "merged_rows_before_dedup": int(len(merged) + dropped_duplicate_rows),
        "dropped_duplicate_rows": dropped_duplicate_rows,
        "merged_rows": int(len(merged)),
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "all_timestamps_utc": True,
        "all_on_5m_grid": True,
        "duplicate_timestamps_after_merge": int(merged.index.duplicated().sum()),
        "existing_output": str(OUTPUT.relative_to(ROOT)),
        "existing_comparison": comparison,
        "output": str(OUTPUT.relative_to(ROOT)),
    }

    print(f"price_basis_status: {basis_status}")
    print(f"price_basis_flagged: {basis_flagged}")
    for row in source_rows:
        print(
            f"source: {row['path']} rows={row['raw_rows']} "
            f"duplicates={row['duplicates_within_source']} "
            f"basis={row['price_basis_hint']} "
            f"{row['first_timestamp']} -> {row['last_timestamp']}"
        )
    for row in overlap_rows:
        print(
            f"overlap: {row['left']} / {row['right']} rows={row['overlap_rows']}"
        )
    print(f"merged_rows_before_dedup: {report['merged_rows_before_dedup']}")
    print(f"dropped_duplicate_rows: {dropped_duplicate_rows}")
    print(f"merged_rows: {len(merged)}")
    print(f"first_timestamp: {first}")
    print(f"last_timestamp: {last}")
    print(f"existing_identical: {comparison.get('identical')}")

    if not bool(comparison.get("identical")):
        _iso_z(merged).to_csv(CANDIDATE, index=False)
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise ValueError(
            "merged data does not match existing full file; "
            f"candidate written to {CANDIDATE}"
        )

    _iso_z(merged).to_csv(OUTPUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {OUTPUT}")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
