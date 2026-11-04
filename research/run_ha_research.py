"""Run the frozen Heikin Ashi swing research.

The in-sample stage never computes or prints out-of-sample results. Run that
stage first, review it, then run the out-of-sample stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_data import CsvMarketDataProvider
from research.ha_swing import run_research

DATA_FILE = Path("data/XAUUSD_5m.csv")
RULES_FILE = Path("research/ha_swing_rules.md")
OUT_FILE = Path("research/ha_swing_report.json")


def _format(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _print_metadata(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    print(
        f"DATA {metadata['data_file']} "
        f"{metadata['actual_start']} -> {metadata['actual_end']}"
    )
    print(
        f"BARS five_min={metadata['five_min_rows']} "
        f"flat_removed={metadata['flat_padding_removed']} "
        f"complete_1h={metadata['complete_hourly_bars']} "
        f"incomplete_dropped={metadata['incomplete_hours_dropped']}"
    )
    print(
        f"SPLIT unique_days={metadata['unique_days']} "
        f"in_sample_days={metadata['in_sample_days']} "
        f"in_sample={metadata['in_sample_start']} -> {metadata['in_sample_end']}"
    )
    if "out_of_sample_days" in metadata:
        print(
            f"SPLIT out_of_sample_days={metadata['out_of_sample_days']} "
            f"out_of_sample={metadata['out_of_sample_start']} -> "
            f"{metadata['out_of_sample_end']}"
        )
    for note in metadata.get("data_notes", []):
        print(f"NOTE {note}")


def _print_metrics(label: str, block: dict[str, Any]) -> None:
    for name in ("all", "long", "short"):
        metrics = block[name]
        print(
            f"{label} {name.upper()} trades={metrics['trades']} "
            f"win_rate_10bp={metrics['win_rate_10bp']:.2f}% "
            f"win_rate_gross={metrics['win_rate_gross']:.2f}% "
            f"avg_gross_R={metrics['avg_gross_r']:.4f} "
            f"expectancy_10bp={metrics['expectancy_10bp']:.4f}R "
            f"expectancy_20bp={metrics['expectancy_20bp']:.4f}R "
            f"worst_10bp={_format(metrics['worst_trade_10bp'])}R "
            f"worst_gross={_format(metrics['worst_trade_gross'])}R "
            f"max_dd_10bp={metrics['max_drawdown_10bp']:.4f}R "
            f"max_dd_20bp={metrics['max_drawdown_20bp']:.4f}R"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("in-sample", "out-of-sample", "all"),
        default="all",
        help="in-sample review, out-of-sample review, or both",
    )
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    if not RULES_FILE.exists():
        raise SystemExit(f"missing frozen rules file: {RULES_FILE}")

    provider = CsvMarketDataProvider(files={"XAUUSD": args.data_file})
    five_min = provider.get_bars("XAUUSD", timeframe="5Min")

    if args.stage == "in-sample":
        report = run_research(five_min, include_out_of_sample=False)
        _print_metadata(report)
        print("IN-SAMPLE")
        _print_metrics("IN-SAMPLE", report["in_sample"]["metrics"])
        print("OUT-OF-SAMPLE LOCKED: not computed in this stage")
        return

    report = run_research(five_min, include_out_of_sample=True)
    _print_metadata(report)
    if args.stage == "all":
        print("IN-SAMPLE")
        _print_metrics("IN-SAMPLE", report["in_sample"]["metrics"])
    print("OUT-OF-SAMPLE")
    _print_metrics("OUT-OF-SAMPLE", report["out_of_sample"]["metrics"])
    print(
        "BUY-AND-HOLD OOS "
        f"{report['out_of_sample']['buy_and_hold_return_pct']:.4f}%"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
