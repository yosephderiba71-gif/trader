"""Run fixed SDT2 variants A and B on the two new XAUUSD files.

Read-only research. No parameter tuning and no variant combinations.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from research.london_sdt import (
    SdtConfig,
    clean_bars,
    load_bars,
    run_sdt,
    summarize,
    variant_config,
)

FILES = {
    "2020-2022": Path("/Users/yosi/download/xauusd-m5-bid-2020-01-01-2023-01-01.csv"),
    "2025-2026": Path("/Users/yosi/download/xauusd-m5-bid-2025-01-01-2026-10-31.csv"),
}
VARIANTS = ("A", "B")
COST_R = 0.05
OUT = Path("research/london_sdt_new_files_AB.json")


def _flat_mask(bars):
    return (
        (bars["open"] == bars["high"])
        & (bars["high"] == bars["low"])
        & (bars["low"] == bars["close"])
    )


def _session_count(bars, config: SdtConfig, year: int) -> int:
    session = bars[
        (bars.index.time >= config.session_start)
        & (bars.index.time < config.session_end)
    ]
    return len({day for day in session.index.date if day.year == year})


def _yearly(trades, bars, config: SdtConfig, years: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for year in years:
        year_trades = [trade for trade in trades if trade.day.startswith(str(year))]
        out[str(year)] = summarize(
            year_trades,
            sessions=_session_count(bars, config, year),
        )
    return out


def _build(label: str, path: Path) -> dict[str, Any]:
    raw = load_bars(path)
    bars = clean_bars(raw)
    flat_count = int(_flat_mask(raw).sum())
    years = sorted({day.year for day in bars.index.date})
    variants: dict[str, Any] = {}
    for name in VARIANTS:
        config = replace(variant_config(name), cost_r=COST_R)
        result = run_sdt(raw, config, data_source=str(path), clean=True)
        variants[name] = {
            "settings": result.report()["settings"],
            "overall": summarize(result.trades, sessions=result.sessions),
            "yearly": _yearly(result.trades, bars, config, years),
            "trades": result.report()["trades"],
        }
    return {
        "label": label,
        "path": str(path),
        "cost_r": COST_R,
        "cost_included": True,
        "cost_model": "net_r = gross_r - cost_r for every trade",
        "cleaning": {
            "applied": True,
            "raw_bars": len(raw),
            "flat_bars_dropped": flat_count,
            "closed_market_bars_dropped": len(raw) - flat_count - len(bars),
            "clean_bars": len(bars),
            "clean_first": str(bars.index.min()),
            "clean_last": str(bars.index.max()),
            "closed_market_rule": (
                "Saturday, Sunday before 22:00 UTC, and Friday at/after 22:00 UTC"
            ),
            "prior_day_rule": (
                "most recent prior UTC date with real bars; Sundays excluded"
            ),
        },
        "years": years,
        "variants": variants,
    }


def _print(report: dict[str, Any]) -> None:
    print(f"FILE {report['label']} {report['path']}")
    print(
        f"  raw={report['cleaning']['raw_bars']} "
        f"flat_dropped={report['cleaning']['flat_bars_dropped']} "
        f"clean={report['cleaning']['clean_bars']} "
        f"range={report['cleaning']['clean_first']} -> {report['cleaning']['clean_last']}"
    )
    for name in VARIANTS:
        variant = report["variants"][name]
        m = variant["overall"]
        print(
            f"  {name} overall trades={m['trades']} win_rate={m['win_rate']:.4f} "
            f"avg_win={m['avg_win_r']:.4f} avg_loss={m['avg_loss_r']:.4f} "
            f"expectancy={m['expectancy_r']:.4f} pf={m['profit_factor']:.4f} "
            f"max_dd={m['max_drawdown_r']:.4f}"
        )
        for year, ym in variant["yearly"].items():
            print(
                f"    {year} trades={ym['trades']} win_rate={ym['win_rate']:.4f} "
                f"avg_win={ym['avg_win_r']:.4f} avg_loss={ym['avg_loss_r']:.4f} "
                f"expectancy={ym['expectancy_r']:.4f} pf={ym['profit_factor']:.4f} "
                f"max_dd={ym['max_drawdown_r']:.4f}"
            )


def main() -> int:
    reports = [_build(label, path) for label, path in FILES.items()]
    OUT.write_text(
        json.dumps({"files": reports}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for report in reports:
        _print(report)
    print(f"report={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
