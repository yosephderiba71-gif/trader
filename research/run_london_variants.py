"""Generate one fixed-cost report for each requested SDT2 variant.

Read-only research. Variants are fixed and intentionally not combined.
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

CSV = Path("data/XAUUSD_5m.csv")
OUT_DIR = Path("research")
COST_R = 0.05
YEARS = (2023, 2024)
VARIANTS = {
    "A": "baseline",
    "B": "skip day when prior-day range is outside 0.30%-1.60% of prior close",
    "C": "trade only in the direction of the 20-trading-day close trend",
    "D": "no trailing stop; fixed stop and 2R target",
    "E": "trail only after +1R; ratchet to the prior closed candle",
}


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


def _yearly(trades, bars, config: SdtConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for year in YEARS:
        year_trades = [trade for trade in trades if trade.day.startswith(str(year))]
        out[str(year)] = summarize(
            year_trades,
            sessions=_session_count(bars, config, year),
        )
    return out


def _cleaning(raw, bars) -> dict[str, Any]:
    flat_count = int(_flat_mask(raw).sum())
    return {
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
        "prior_day_rule": "most recent prior UTC date with real bars; Sundays excluded",
    }


def _build_variant(name: str, raw, bars) -> dict[str, Any]:
    config = replace(variant_config(name), cost_r=COST_R)
    result = run_sdt(raw, config, data_source=str(CSV), clean=True)
    report = result.report()
    report.update(
        {
            "variant": name,
            "variant_description": VARIANTS[name],
            "cost_r": COST_R,
            "cost_included": True,
            "cost_model": "net_r = gross_r - cost_r for every trade",
            "cleaning": _cleaning(raw, bars),
            "yearly": _yearly(result.trades, bars, config),
        }
    )
    return report


def _print_variant(report: dict[str, Any]) -> None:
    m = report["metrics"]
    print(
        f"{report['variant']}: trades={m['trades']} win_rate={m['win_rate']:.4f} "
        f"avg_win={m['avg_win_r']:.4f} avg_loss={m['avg_loss_r']:.4f} "
        f"expectancy={m['expectancy_r']:.4f} pf={m['profit_factor']} "
        f"max_dd={m['max_drawdown_r']:.4f}"
    )
    for year, ym in report["yearly"].items():
        print(
            f"  {year}: trades={ym['trades']} win_rate={ym['win_rate']:.4f} "
            f"avg_win={ym['avg_win_r']:.4f} avg_loss={ym['avg_loss_r']:.4f} "
            f"expectancy={ym['expectancy_r']:.4f} pf={ym['profit_factor']} "
            f"max_dd={ym['max_drawdown_r']:.4f}"
        )


def main() -> int:
    raw = load_bars(CSV)
    bars = clean_bars(raw)
    for name in VARIANTS:
        report = _build_variant(name, raw, bars)
        path = OUT_DIR / f"london_sdt_variant_{name}.json"
        path.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        _print_variant(report)
        print(f"report={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
