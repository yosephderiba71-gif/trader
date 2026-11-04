"""Run fixed gold variants I, J, and K.

Read-only research. No parameter tuning and no variant combinations.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from research.gold_variants import (
    DailyJConfig,
    DailyKConfig,
    ReversionConfig,
    reversion_with_cost,
    run_daily_j,
    run_daily_k,
    run_reversion,
    summarize_daily_trades,
    summarize_reversion,
)
from research.london_sdt import clean_bars, load_bars

FILES = [
    ("2020-2022", Path("/Users/yosi/download/xauusd-m5-bid-2020-01-01-2023-01-01.csv")),
    ("2023-2024", Path("data/XAUUSD_5m.csv")),
    ("2025-2026", Path("/Users/yosi/download/xauusd-m5-bid-2025-01-01-2026-10-31.csv")),
]
DAILY_PATH = Path("data/gold_daily_gc_f_2010.csv")
DAILY_SOURCE = "Yahoo Finance GC=F daily, downloaded 2026-11-03"
COSTS = (0.0, 0.05, 0.10)
INTRADAY_YEARS = tuple(range(2020, 2027))
DAILY_YEARS = tuple(range(2010, 2027))
OUT = Path("research/gold_variants_report.json")


def _flat_mask(bars: pd.DataFrame) -> pd.Series:
    return (
        (bars["open"] == bars["high"])
        & (bars["high"] == bars["low"])
        & (bars["low"] == bars["close"])
    )


def _intraday_yearly(trades) -> dict[str, Any]:
    return {
        str(year): summarize_reversion(
            [trade for trade in trades if trade.day.startswith(str(year))]
        )
        for year in INTRADAY_YEARS
    }


def _daily_yearly(trades) -> dict[str, Any]:
    return {
        str(year): summarize_daily_trades(
            [trade for trade in trades if trade.entry_date.startswith(str(year))]
        )
        for year in DAILY_YEARS
    }


def _print_reversion(report: dict[str, Any]) -> None:
    print(f"VARIANT I cost={report['cost_r']:.2f}R")
    for year, metrics in report["yearly"].items():
        print(
            f"  {year} trades={metrics['trades']} "
            f"win_rate={metrics['win_rate']:.4f} "
            f"avg_win={metrics['avg_win_r']:.4f} "
            f"avg_loss={metrics['avg_loss_r']:.4f} "
            f"worst={metrics['worst_loss_r']:.4f} "
            f"expectancy={metrics['expectancy_r']:.4f} "
            f"pf={metrics['profit_factor']:.4f}"
        )
    m = report["metrics"]
    print(
        f"  TOTAL trades={m['trades']} win_rate={m['win_rate']:.4f} "
        f"avg_win={m['avg_win_r']:.4f} avg_loss={m['avg_loss_r']:.4f} "
        f"worst={m['worst_loss_r']:.4f} expectancy={m['expectancy_r']:.4f} "
        f"pf={m['profit_factor']:.4f}"
    )


def _print_daily_j(report: dict[str, Any]) -> None:
    print("VARIANT J")
    for year, metrics in report["yearly"].items():
        print(
            f"  {year} trades={metrics['trades']} "
            f"win_rate={metrics['win_rate']:.4f} "
            f"avg_win={metrics['avg_win_pct']:.4f}% "
            f"avg_loss={metrics['avg_loss_pct']:.4f}% "
            f"worst={metrics['worst_loss_pct']:.4f}% "
            f"expectancy={metrics['expectancy_pct']:.4f}% "
            f"pf={metrics['profit_factor']:.4f} "
            f"total={metrics['total_return_pct']:.4f}%"
        )
    m = report["metrics"]
    print(
        f"  TOTAL trades={m['trades']} win_rate={m['win_rate']:.4f} "
        f"avg_win={m['avg_win_pct']:.4f}% avg_loss={m['avg_loss_pct']:.4f}% "
        f"worst={m['worst_loss_pct']:.4f}% expectancy={m['expectancy_pct']:.4f}% "
        f"pf={m['profit_factor']:.4f} total={m['total_return_pct']:.4f}%"
    )


def _print_daily_k(report: dict[str, Any]) -> None:
    print("VARIANT K")
    for year, metrics in report["yearly"].items():
        print(
            f"  {year} entries={metrics['entries']} "
            f"days_long={metrics['days_invested']} "
            f"strategy={metrics['strategy_return_pct']:.4f}% "
            f"buy_hold={metrics['buy_hold_return_pct']:.4f}%"
        )
    m = report["metrics"]
    print(
        f"  TOTAL entries={m['entries']} days_long={m['days_invested']} "
        f"strategy={m['strategy_return_pct']:.4f}% "
        f"buy_hold={m['buy_hold_return_pct']:.4f}% "
        f"max_dd={m['strategy_max_drawdown_pct']:.4f}%"
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
    }
    print(
        f"ALL raw={cleaning['raw_bars']} flat_dropped={flat_count} "
        f"clean={len(bars)} duplicates={duplicates} "
        f"range={bars.index.min()} -> {bars.index.max()}"
    )

    base = run_reversion(
        bars,
        ReversionConfig(cost_r=0.0),
        data_source="2020-2022 + 2023-2024 + 2025-2026",
        clean=False,
    )
    print(f"VARIANT I signals_to_trades={len(base.trades)}")
    runs = []
    for cost in COSTS:
        trades = reversion_with_cost(base, cost)
        report = {
            "variant": "I",
            "cost_r": cost,
            "cost_included": True,
            "cost_model": "net_r = gross_r - cost_r for every trade",
            "settings": asdict(replace(base.config, cost_r=cost)),
            "metrics": summarize_reversion(trades),
            "yearly": _intraday_yearly(trades),
            "trades": [asdict(trade) for trade in trades],
        }
        runs.append(report)
        _print_reversion(report)

    daily = load_bars(DAILY_PATH)
    ohlc_violations = int(
        (
            (daily["high"] < daily[["open", "close"]].max(axis=1))
            | (daily["low"] > daily[["open", "close"]].min(axis=1))
        ).sum()
    )
    daily_quality = {
        "source": DAILY_SOURCE,
        "symbol": "GC=F",
        "rows": len(daily),
        "first": str(daily.index.min()),
        "last": str(daily.index.max()),
        "duplicate_dates": int(daily.index.duplicated().sum()),
        "null_ohlc": int(daily[["open", "high", "low", "close"]].isna().sum().sum()),
        "ohlc_envelope_violations": ohlc_violations,
    }
    print(
        f"DAILY {DAILY_SOURCE} rows={len(daily)} "
        f"range={daily.index.min()} -> {daily.index.max()} "
        f"ohlc_violations={ohlc_violations}"
    )

    j_result = run_daily_j(daily, DailyJConfig(), data_source=DAILY_SOURCE)
    j_report = j_result.report()
    j_report["yearly"] = _daily_yearly(j_result.trades)
    _print_daily_j(j_report)

    k_result = run_daily_k(daily, DailyKConfig(), data_source=DAILY_SOURCE)
    k_report = k_result.report()
    _print_daily_k(k_report)

    OUT.write_text(
        json.dumps(
            {
                "strategy": "Gold variants I/J/K",
                "intraday": {
                    "data_sources": [str(path) for _, path in FILES],
                    "cleaning": cleaning,
                    "costs_r": list(COSTS),
                    "runs": runs,
                },
                "daily": {
                    "source": DAILY_SOURCE,
                    "quality": daily_quality,
                    "costs": "none specified; J/K reported as gross price returns",
                    "j": j_report,
                    "k": k_report,
                },
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
