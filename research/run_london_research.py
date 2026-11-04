"""Run the SDT² London breakout simulator on local 5-minute gold bars.

Read-only research. This runner never imports a broker, never reads API keys,
and never places orders.

Examples:
    python -m research.run_london_research
    python -m research.run_london_research --synthetic
    python -m research.run_london_research --csv data/XAUUSD_5m.csv --out research/london_sdt_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.london_sdt import SdtConfig, load_bars, make_synthetic_bars, run_sdt

DEFAULT_CSV = Path("data/XAUUSD_5m.csv")
DEFAULT_OUT = Path("research/london_sdt_report.json")


def _date(value: str) -> date:
    return date.fromisoformat(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDT² London breakout research simulator")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="local 5-minute OHLCV CSV")
    parser.add_argument("--synthetic", action="store_true", help="use a deterministic synthetic fixture")
    parser.add_argument("--synthetic-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON report path")
    parser.add_argument("--target-r", type=float, default=2.0)
    parser.add_argument("--cost-r", type=float, default=0.05)
    parser.add_argument("--start", type=_date, default=None)
    parser.add_argument("--end", type=_date, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic:
        bars = make_synthetic_bars(days=args.synthetic_days, seed=args.seed)
        source = "synthetic"
        warning = "Synthetic fixture; not a performance claim."
    else:
        path = Path(args.csv)
        if not path.exists():
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{path} not found. Provide the local CSV or run with --synthetic.",
                    },
                    indent=2,
                )
            )
            return 2
        bars = load_bars(path)
        source = str(path)
        warning = "Local CSV; research only; no live trading."

    config = SdtConfig(target_r=args.target_r, cost_r=args.cost_r)
    result = run_sdt(
        bars,
        config,
        start=args.start,
        end=args.end,
        data_source=source,
    )
    payload = {"ok": True, "warning": warning, **result.report()}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "data_source": source,
                "warning": warning,
                "metrics": payload["metrics"],
                "report": str(target),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
