"""Walk-forward check of the opening-range breakout rules on historical 5-minute bars.

Read-only research. This module imports only the market-data client, never the trading client,
so it cannot place orders. It writes a JSON report to research/results/.

Run: python -m research.run_orb_research
Environment: ALPACA_API_KEY, ALPACA_API_SECRET (read only, never printed), optional
SYMBOLS (default GLD,SPY) and LOOKBACK_DAYS (default 730; use 1825 for five years).

Important: these gates were set after the first two-year run, and the same history is reused.
A pass here is a filter, not proof of an edge. The real test is trading on days that come after
the rules are fixed.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from trader.orb import OrbConfig, run_orb, summarize

TRAIN_FRACTION = 0.7
GATES = {
    "min_oos_trades_per_symbol": 100,
    "min_oos_expectancy_r": 0.10,            # at the normal 0.05R cost assumption
    "min_oos_expectancy_double_cost_r": 0.0,  # still not losing at 0.10R costs
    "min_positive_halves": 2,                # both halves of all the data must be positive
    "min_years": 5,                          # need at least five calendar years of data
    "min_positive_years": 4,                 # at least four of those years must be positive
    "max_oos_drawdown_r": -10.0,
}
DOUBLE_COST_R = 0.10
VARIANTS = {"trail": OrbConfig(trail=True), "fixed_stop": OrbConfig(trail=False)}
OUT_DIR = Path("research/results")


def load_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """5-minute OHLC bars from Alpaca's IEX feed, with a UTC index. Empty if none came back."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
    client = StockHistoricalDataClient(key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    raw = client.get_stock_bars(request).df
    if raw.empty:
        return pd.DataFrame()
    return raw.xs(symbol, level=0)[["open", "high", "low", "close"]].sort_index()


def split_by_day(df: pd.DataFrame, cfg: OrbConfig, train_fraction: float = TRAIN_FRACTION):
    """Earlier trading days for in-sample, later days for out-of-sample. Split by whole days."""
    days_local = df.index.tz_convert(cfg.tz).date
    days = sorted(set(days_local))
    cut = days[int(len(days) * train_fraction)]
    return df[days_local < cut], df[days_local >= cut]


def evaluate(symbol: str, df: pd.DataFrame, variant: str, cfg: OrbConfig) -> dict:
    train, test = split_by_day(df, cfg)
    in_sample = summarize(run_orb(train, cfg))
    out_sample = summarize(run_orb(test, cfg))
    double_cost = summarize(run_orb(test, replace(cfg, cost_r=DOUBLE_COST_R)))

    first_half, second_half = split_by_day(df, cfg, train_fraction=0.5)
    halves = [summarize(run_orb(first_half, cfg)), summarize(run_orb(second_half, cfg))]

    all_trades = run_orb(df, cfg)
    years = {y: summarize([t for t in all_trades if t.day[:4] == y])
             for y in sorted({t.day[:4] for t in all_trades})}

    positive_halves = sum(1 for h in halves if h["trades"] > 0 and h["expectancy_r"] > 0)
    positive_years = sum(1 for y in years.values() if y["trades"] > 0 and y["expectancy_r"] > 0)
    checks = {
        "oos_trades": out_sample["trades"] >= GATES["min_oos_trades_per_symbol"],
        "oos_expectancy": out_sample["expectancy_r"] >= GATES["min_oos_expectancy_r"],
        "oos_expectancy_double_cost": double_cost["expectancy_r"] >= GATES["min_oos_expectancy_double_cost_r"],
        "both_halves_positive": positive_halves >= GATES["min_positive_halves"],
        "enough_years": len(years) >= GATES["min_years"],
        "positive_years": positive_years >= GATES["min_positive_years"],
        "oos_drawdown": out_sample["max_drawdown_r"] >= GATES["max_oos_drawdown_r"],
    }
    return {
        "symbol": symbol,
        "variant": variant,
        "in_sample": in_sample,
        "out_of_sample": out_sample,
        "out_of_sample_double_cost": double_cost,
        "halves": halves,
        "per_year": years,
        "checks": checks,
        "passed_gates": all(checks.values()),
    }


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(os.getenv("LOOKBACK_DAYS", "730")))
    symbols = [s.strip() for s in os.getenv("SYMBOLS", "GLD,SPY").split(",") if s.strip()]
    report = {
        "run_at": end.isoformat(timespec="seconds"),
        "gates": GATES,
        "train_fraction": TRAIN_FRACTION,
        "results": [],
    }
    for symbol in symbols:
        df = load_bars(symbol, start, end)
        for variant, cfg in VARIANTS.items():
            if df.empty:
                report["results"].append({"symbol": symbol, "variant": variant, "error": "no bars returned"})
                continue
            report["results"].append(evaluate(symbol, df, variant, cfg))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"orb_{end:%Y%m%d_%H%M}.json"
    text = json.dumps(report, indent=2, default=str)
    path.write_text(text)
    print(text)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
