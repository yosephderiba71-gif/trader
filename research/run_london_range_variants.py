"""Run fixed range variants F, G, and H on all XAUUSD data.

Read-only research. No parameter tuning and no variant combinations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.london_range_breakout import (
    RangeConfig,
    range_variant_config,
    run_range_variant,
)
from research.london_sdt import clean_bars, load_bars, summarize

FILES = [
    ("2020-2022", Path("/Users/yosi/download/xauusd-m5-bid-2020-01-01-2023-01-01.csv")),
    ("2023-2024", Path("data/XAUUSD_5m.csv")),
    ("2025-2026", Path("/Users/yosi/download/xauusd-m5-bid-2025-01-01-2026-10-31.csv")),
]
VARIANTS = ("F", "G", "H")
COSTS = (0.0, 0.05, 0.10)
YEARS = tuple(range(2020, 2027))
OUT = Path("research/london_range_variants_report.json")


def _flat_mask(bars):
    return (
        (bars["open"] == bars["high"])
        & (bars["high"] == bars["low"])
        & (bars["low"] == bars["close"])
    )


def _yearly(trades, config: RangeConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for year in YEARS:
        year_trades = [trade for trade in trades if trade.day.startswith(str(year))]
        out[str(year)] = summarize(year_trades)
    return out


def _build_run(bars, name: str, cost: float) -> dict[str, Any]:
    config = range_variant_config(name, cost_r=cost)
    result = run_range_variant(
        bars,
        config,
        data_source="2020-2022 + 2023-2024 + 2025-2026",
        clean=False,
    )
    report = result.report()
    report.update(
        {
            "variant": name,
            "cost_r": cost,
            "cost_included": True,
            "cost_model": "net_r = gross_r - cost_r for every trade",
            "yearly": _yearly(result.trades, config),
        }
    )
    return report


def _print_run(report: dict[str, Any]) -> None:
    print(f"VARIANT {report['variant']} cost={report['cost_r']:.2f}R")
    for year, metrics in report["yearly"].items():
        print(
            f"  {year} trades={metrics['trades']} "
            f"win_rate={metrics['win_rate']:.4f} "
            f"expectancy={metrics['expectancy_r']:.4f} "
            f"pf={metrics['profit_factor']:.4f}"
        )
    m = report["metrics"]
    print(
        f"  TOTAL trades={m['trades']} win_rate={m['win_rate']:.4f} "
        f"expectancy={m['expectancy_r']:.4f} pf={m['profit_factor']:.4f}"
    )


def main() -> int:
    raw_frames = []
    clean_frames = []
    for label, path in FILES:
        raw = load_bars(path)
        clean = clean_bars(raw)
        raw_frames.append(raw)
        clean_frames.append(clean)
        print(
            f"FILE {label} {path} raw={len(raw)} clean={len(clean)} "
            f"range={clean.index.min()} -> {clean.index.max()}"
        )
    raw = pd.concat(raw_frames).sort_index()
    bars = pd.concat(clean_frames).sort_index()
    duplicates = int(bars.index.duplicated().sum())
    if duplicates:
        raise ValueError(f"duplicate timestamps across files: {duplicates}")
    flat_count = int(_flat_mask(raw).sum())
    cleaning = {
        "applied": True,
        "raw_bars": len(raw),
        "flat_bars_dropped": flat_count,
        "closed_market_bars_dropped": len(raw) - flat_count - len(bars),
        "clean_bars": len(bars),
        "clean_first": str(bars.index.min()),
        "clean_last": str(bars.index.max()),
        "duplicate_timestamps": duplicates,
        "closed_market_rule": (
            "Saturday, Sunday before 22:00 UTC, and Friday at/after 22:00 UTC"
        ),
        "prior_day_rule": (
            "most recent prior UTC date with real bars; Sundays excluded"
        ),
    }
    print(
        f"ALL raw={cleaning['raw_bars']} flat_dropped={flat_count} "
        f"clean={len(bars)} duplicates={duplicates} "
        f"range={bars.index.min()} -> {bars.index.max()}"
    )
    runs = []
    for name in VARIANTS:
        for cost in COSTS:
            report = _build_run(bars, name, cost)
            runs.append(report)
            _print_run(report)
    OUT.write_text(
        json.dumps(
            {
                "strategy": "Range variants F/G/H",
                "data_sources": [str(path) for _, path in FILES],
                "costs_r": list(COSTS),
                "cost_included": True,
                "cost_model": "net_r = gross_r - cost_r for every trade",
                "cleaning": cleaning,
                "runs": runs,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"report={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
