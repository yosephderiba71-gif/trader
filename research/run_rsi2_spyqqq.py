"""Run the frozen RSI(2) SPY/QQQ backtest and emit the report JSON.

Read-only research. The runner never tunes parameters and never combines
variants; it only varies the requested cost assumptions and reports pass
criteria as frozen.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.rsi2_spyqqq import (
    BASE_COST_BPS,
    BASELINE_PERCENTILE,
    HALF_FRACTION,
    STRESS_COST_BPS,
    TRAIN_FRACTION,
    load_daily_bars,
    random_baseline,
    simulate,
    summarize,
)

SYMBOLS = ("SPY", "QQQ")
REPORT_PATH = Path("research/rsi2_spyqqq_report.json")


def _entry_day(bars: pd.DataFrame, trade: Any) -> pd.Timestamp:
    return bars.index[trade.entry_index].normalize()


def _evaluate_symbol(symbol: str, bars: pd.DataFrame) -> dict[str, Any]:
    trades = simulate(bars, symbol=symbol)
    day_index = pd.DatetimeIndex(sorted(pd.DatetimeIndex(bars.index).normalize().unique()))
    n_days = int(len(day_index))
    n_train = max(1, min(int(n_days * TRAIN_FRACTION), n_days - 1))
    train_days = day_index[:n_train]
    oos_days = day_index[n_train:]
    mid = max(1, min(int(n_days * HALF_FRACTION), n_days - 1))
    half_days = (day_index[:mid], day_index[mid:])

    def in_days(trade: Any, days: pd.DatetimeIndex) -> bool:
        return _entry_day(bars, trade) in days

    is_trades = [trade for trade in trades if in_days(trade, train_days)]
    oos_trades = [trade for trade in trades if in_days(trade, oos_days)]
    half_one_trades = [trade for trade in trades if _entry_day(bars, trade) in half_days[0]]
    half_two_trades = [trade for trade in trades if _entry_day(bars, trade) in half_days[1]]

    full_metrics = summarize(trades, BASE_COST_BPS)
    full_metrics_stress = summarize(trades, STRESS_COST_BPS)
    is_metrics = summarize(is_trades, BASE_COST_BPS)
    is_metrics_stress = summarize(is_trades, STRESS_COST_BPS)
    oos_metrics = summarize(oos_trades, BASE_COST_BPS)
    oos_metrics_stress = summarize(oos_trades, STRESS_COST_BPS)

    oos_entry_indices = np.array(
        [float(trade.entry_index) for trade in oos_trades if trade.exit_index is not None],
        dtype=float,
    )
    oos_held = np.array(
        [float(trade.held_days) for trade in oos_trades if trade.exit_index is not None],
        dtype=float,
    )
    baseline = random_baseline(bars, oos_entry_indices, oos_held, BASE_COST_BPS)

    half_one_metrics = summarize(half_one_trades, BASE_COST_BPS)
    half_two_metrics = summarize(half_two_trades, BASE_COST_BPS)

    years: dict[str, list[Any]] = {}
    for trade in trades:
        year = str(_entry_day(bars, trade).year)
        years.setdefault(year, []).append(trade)
    yearly: dict[str, Any] = {}
    positive_years = 0
    for year in sorted(years):
        year_trades = years[year]
        metrics = summarize(year_trades, BASE_COST_BPS)
        yearly[year] = {
            "trades": metrics["trades"],
            "win_rate": metrics["win_rate"],
            "expectancy_pct": metrics["expectancy_pct"],
            "profit_factor": metrics["profit_factor"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
        }
        if metrics["expectancy_pct"] > 0:
            positive_years += 1
    year_fraction = (positive_years / len(years)) if years else 0.0

    passes = {
        "P1": bool(oos_metrics_stress["expectancy_pct"] > 0.0),
        "P2": bool(oos_metrics["expectancy_pct"] >= 0.20),
        "P3": bool(len(trades) >= 100),
        "P4": bool(oos_metrics["expectancy_pct"] > baseline["baseline_pct"]),
        "P5": bool(half_one_metrics["expectancy_pct"] > 0.0 and half_two_metrics["expectancy_pct"] > 0.0),
        "P6": bool(oos_metrics["max_drawdown_pct"] >= -20.0),
        "P7": bool(year_fraction >= 0.60),
    }

    return {
        "symbol": symbol,
        "data": {
            "first_day": str(day_index[0].date()) if n_days else None,
            "last_day": str(day_index[-1].date()) if n_days else None,
            "trading_days": n_days,
            "in_sample_days": int(n_train),
            "in_sample_start": str(train_days[0].date()),
            "in_sample_end": str(train_days[-1].date()),
            "out_of_sample_start": str(oos_days[0].date()),
            "out_of_sample_end": str(oos_days[-1].date()),
        },
        "full_history": full_metrics,
        "full_history_stress": full_metrics_stress,
        "in_sample": {
            "trades": is_metrics["trades"],
            "win_rate": is_metrics["win_rate"],
            "expectancy_pct": is_metrics["expectancy_pct"],
            "profit_factor": is_metrics["profit_factor"],
            "max_drawdown_pct": is_metrics["max_drawdown_pct"],
        },
        "in_sample_stress": {
            "trades": is_metrics_stress["trades"],
            "expectancy_pct": is_metrics_stress["expectancy_pct"],
            "max_drawdown_pct": is_metrics_stress["max_drawdown_pct"],
        },
        "out_of_sample": {
            "trades": oos_metrics["trades"],
            "win_rate": oos_metrics["win_rate"],
            "expectancy_pct": oos_metrics["expectancy_pct"],
            "profit_factor": oos_metrics["profit_factor"],
            "max_drawdown_pct": oos_metrics["max_drawdown_pct"],
        },
        "out_of_sample_stress": {
            "trades": oos_metrics_stress["trades"],
            "expectancy_pct": oos_metrics_stress["expectancy_pct"],
            "max_drawdown_pct": oos_metrics_stress["max_drawdown_pct"],
        },
        "out_of_sample_baseline": {
            "runs": baseline["runs"],
            "percentile": baseline["baseline_pct"],
            "mean_pct": baseline["expectancy_pct"],
        },
        "halves": {
            "first": {
                "trades": half_one_metrics["trades"],
                "expectancy_pct": half_one_metrics["expectancy_pct"],
            },
            "second": {
                "trades": half_two_metrics["trades"],
                "expectancy_pct": half_two_metrics["expectancy_pct"],
            },
        },
        "calendar_years": yearly,
        "pass_fraction_years_positive": float(year_fraction),
        "passes": passes,
        "trades": [asdict(trade) for trade in trades],
    }


def main() -> int:
    reports: dict[str, Any] = {}
    for symbol in SYMBOLS:
        start = "2000-01-01" if symbol == "SPY" else None
        bars = load_daily_bars(symbol, start=start)
        reports[symbol] = _evaluate_symbol(symbol, bars)
        report = reports[symbol]
        oos = report["out_of_sample"]
        stress = report["out_of_sample_stress"]
        baseline = report["out_of_sample_baseline"]
        print(
            f"{symbol}: trading_days={report['data']['trading_days']} "
            f"full_trades={report['full_history']['trades']} "
            f"oos_trades={oos['trades']} "
            f"oos_expectancy_base={oos['expectancy_pct']:.4f} "
            f"oos_expectancy_stress={stress['expectancy_pct']:.4f} "
            f"oos_max_dd={oos['max_drawdown_pct']:.4f} "
            f"baseline_95th={baseline['percentile']:.4f} "
            f"passes={report['passes']}"
        )
    REPORT_PATH.write_text(json.dumps(reports, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
