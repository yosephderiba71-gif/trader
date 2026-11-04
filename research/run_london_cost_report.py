"""Generate raw and cleaned SDT2 cost-scenario reports.

Read-only research. This script does not tune strategy parameters; it only
varies the requested transaction-cost assumption.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.london_sdt import SdtConfig, clean_bars, load_bars, run_sdt, summarize

CSV = Path("data/XAUUSD_5m.csv")
RAW_OUT = Path("research/london_sdt_report_raw.json")
CLEAN_OUT = Path("research/london_sdt_report_clean.json")
COSTS = (0.0, 0.05, 0.10)
YEARS = (2023, 2024)


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


def _build_report(mode: str, raw, bars_for_run, clean: bool) -> dict[str, Any]:
    flat = _flat_mask(raw)
    clean_frame = clean_bars(raw)
    flat_count = int(flat.sum())
    closed_nonflat_count = len(raw) - flat_count - len(clean_frame)
    cleaning = {
        "applied": clean,
        "raw_bars": len(raw),
        "flat_bars_present": flat_count,
        "closed_market_bars_present": closed_nonflat_count,
        "flat_bars_dropped": flat_count if clean else 0,
        "closed_market_bars_dropped": closed_nonflat_count if clean else 0,
        "clean_bars": len(clean_frame) if clean else None,
        "clean_first": str(clean_frame.index.min()) if clean else None,
        "clean_last": str(clean_frame.index.max()) if clean else None,
        "closed_market_rule": (
            "Saturday, Sunday before 22:00 UTC, and Friday at/after 22:00 UTC"
            if clean
            else "not applied"
        ),
        "prior_day_rule": (
            "most recent prior UTC date with real bars; Sundays excluded"
            if clean
            else "legacy raw behavior; flat padding retained and Sundays may count"
        ),
    }
    runs = []
    for cost in COSTS:
        config = SdtConfig(cost_r=cost)
        result = run_sdt(
            raw,
            config,
            data_source=str(CSV),
            clean=clean,
        )
        report = result.report()
        report["cost_r"] = cost
        report["cost_included"] = True
        report["yearly"] = _yearly(result.trades, bars_for_run, config)
        runs.append(report)
    return {
        "strategy": "SDT2 London breakout",
        "mode": mode,
        "data_source": str(CSV),
        "costs_r": list(COSTS),
        "cost_included": True,
        "cost_model": "net_r = gross_r - cost_r for every trade",
        "cleaning": cleaning,
        "runs": runs,
    }


def _print_summary(mode: str, report: dict[str, Any]) -> None:
    print(f"=== {mode} ===")
    for run in report["runs"]:
        m = run["metrics"]
        print(
            f"cost={run['cost_r']:.2f}R trades={m['trades']} "
            f"win_rate={m['win_rate']:.4f} avg_win={m['avg_win_r']:.4f} "
            f"avg_loss={m['avg_loss_r']:.4f} expectancy={m['expectancy_r']:.4f} "
            f"pf={m['profit_factor']} max_dd={m['max_drawdown_r']:.4f}"
        )
        for year, ym in run["yearly"].items():
            print(
                f"  {year}: trades={ym['trades']} win_rate={ym['win_rate']:.4f} "
                f"avg_win={ym['avg_win_r']:.4f} avg_loss={ym['avg_loss_r']:.4f} "
                f"expectancy={ym['expectancy_r']:.4f} pf={ym['profit_factor']} "
                f"max_dd={ym['max_drawdown_r']:.4f}"
            )


def main() -> int:
    raw = load_bars(CSV)
    clean = clean_bars(raw)
    raw_report = _build_report("raw", raw, raw, clean=False)
    clean_report = _build_report("cleaned", raw, clean, clean=True)

    RAW_OUT.write_text(
        json.dumps(raw_report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    CLEAN_OUT.write_text(
        json.dumps(clean_report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _print_summary("raw", raw_report)
    _print_summary("cleaned", clean_report)
    print(f"raw_report={RAW_OUT}")
    print(f"clean_report={CLEAN_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
