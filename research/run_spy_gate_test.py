"""SPY 200-SMA gate decomposition test.

Rule: hold SPY when SPY Close > SPY 200-day SMA; otherwise hold 100% cash.
Rebalance on the first trading day of each month. Uses the same 70/30
out-of-sample split and 10bp cost model as the Trial #24 sector backtest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.run_sector_momentum import (
    BACKTEST_START,
    BENCHMARK,
    COST_RATE,
    TRAIN_FRACTION,
    cagr,
    calmar_ratio,
    load_prices,
    max_drawdown,
    monthly_rebalances,
    round4,
    sharpe_ratio,
)

SMA_WINDOW = 200
REPORT_PATH = Path("research/spy_gate_report.json")


def gate_decision(spy: pd.Series, index: int) -> tuple[float, float, float]:
    history = spy.iloc[: index + 1].dropna()
    if len(history) < SMA_WINDOW:
        return 0.0, float("nan"), float("nan")
    close = float(history.iloc[-1])
    sma = float(history.iloc[-SMA_WINDOW:].mean())
    target = 1.0 if close > sma else 0.0
    return target, close, sma


def simulate_spy_gate(prices: pd.DataFrame, rebalances: list[int]) -> dict[str, Any]:
    spy = prices[BENCHMARK].ffill()
    returns = spy.pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float)
    n = len(prices)
    rebalance_set = set(rebalances)
    first = rebalances[0]
    weight = 0.0
    equity = 1.0
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    events: list[dict[str, Any]] = []

    def rebalance(index: int, pre: float, current: float) -> tuple[float, float]:
        target, close, sma = gate_decision(spy, index)
        turnover = abs(target - current)
        cost = turnover * COST_RATE
        post_cost_equity = pre * (1.0 - cost)
        events.append(
            {
                "index": int(index),
                "date": prices.index[index],
                "pre_cost_equity": float(pre),
                "post_cost_equity": float(post_cost_equity),
                "turnover": float(turnover),
                "cost": float(cost),
                "target_weight": float(target),
                "spy_close": float(close),
                "spy_sma200": float(sma),
                "invested": bool(target > 0.0),
            }
        )
        return post_cost_equity, target

    equity, weight = rebalance(first, equity, weight)
    dates.append(prices.index[first])
    values.append(float(equity))

    for t in range(first + 1, n):
        r = float(returns[t])
        growth = 1.0 + weight * r
        if growth <= 0:
            raise RuntimeError(f"non-positive equity growth on {prices.index[t]}")
        equity *= growth
        weight = weight * (1.0 + r) / growth
        if t in rebalance_set:
            equity, weight = rebalance(t, equity, weight)
        dates.append(prices.index[t])
        values.append(float(equity))

    return {
        "equity": pd.Series(values, index=pd.DatetimeIndex(dates)),
        "events": events,
    }


def build_report(prices: pd.DataFrame, sim: dict[str, Any]) -> dict[str, Any]:
    equity = sim["equity"]
    events = sim["events"]
    n = len(equity)
    split = max(1, min(int(n * TRAIN_FRACTION), n - 1))
    is_equity = equity.iloc[: split + 1]
    oos_equity = equity.iloc[split:]
    oos_start_date = oos_equity.index[0]
    oos_start = int(prices.index.get_loc(oos_start_date))

    spy = prices[BENCHMARK].ffill()
    spy_oos = float((spy.iloc[-1] / spy.loc[oos_start_date] - 1.0) * 100.0)

    is_total = (is_equity.iloc[-1] / is_equity.iloc[0] - 1.0) * 100.0
    oos_total = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) * 100.0

    return {
        "strategy": "spy_200_sma_gate",
        "rule": "Hold SPY when SPY Close > SPY 200-day SMA; otherwise hold 100% cash.",
        "benchmark": BENCHMARK,
        "cost_bps_per_trade": round4(COST_RATE * 10_000.0),
        "rebalance_frequency": "first_trading_day_of_each_month",
        "data": {
            "source": "yfinance",
            "auto_adjust": True,
            "first_date": str(prices.index[0].date()),
            "last_date": str(prices.index[-1].date()),
            "trading_days": int(len(prices)),
            "backtest_start": str(equity.index[0].date()),
            "backtest_end": str(equity.index[-1].date()),
            "backtest_observations": int(n),
            "is_start": str(is_equity.index[0].date()),
            "is_end": str(is_equity.index[-1].date()),
            "is_daily_returns": int(split),
            "oos_start": str(oos_equity.index[0].date()),
            "oos_end": str(oos_equity.index[-1].date()),
            "oos_daily_returns": int(n - 1 - split),
            "rebalance_count_total": int(len(events)),
            "rebalance_count_is": int(
                sum(1 for event in events if event["index"] < oos_start)
            ),
            "rebalance_count_oos": int(
                sum(1 for event in events if event["index"] >= oos_start)
            ),
        },
        "in_sample": {
            "total_return_pct": round4(is_total),
            "cagr_pct": round4(
                cagr(
                    is_equity.iloc[0],
                    is_equity.iloc[-1],
                    is_equity.index[0],
                    is_equity.index[-1],
                )
            ),
            "max_drawdown_pct": round4(max_drawdown(is_equity)),
            "sharpe_ratio": round4(sharpe_ratio(is_equity)),
            "calmar_ratio": round4(calmar_ratio(is_equity)),
        },
        "out_of_sample": {
            "total_return_pct": round4(oos_total),
            "cagr_pct": round4(
                cagr(
                    oos_equity.iloc[0],
                    oos_equity.iloc[-1],
                    oos_equity.index[0],
                    oos_equity.index[-1],
                )
            ),
            "max_drawdown_pct": round4(max_drawdown(oos_equity)),
            "sharpe_ratio": round4(sharpe_ratio(oos_equity)),
            "calmar_ratio": round4(calmar_ratio(oos_equity)),
            "observations": int(len(oos_equity)),
        },
        "buy_and_hold_spy_oos_return_pct": round4(spy_oos),
        "definitions": {
            "signal_timing": "Signals use SPY closes through the rebalance date; new weights earn the next day's return. No lookahead.",
            "cost_model": "10bp applied to one-way SPY weight turnover; cash earns 0%.",
            "max_drawdown_pct": "Peak-to-trough drawdown as a negative percentage.",
            "calmar_ratio": "CAGR divided by the absolute max drawdown.",
        },
    }


def print_oos_summary(report: dict[str, Any]) -> None:
    oos = report["out_of_sample"]
    print("\nSPY 200-SMA gate OOS summary:")
    print(f"  OOS return: {oos['total_return_pct']:.4f}%")
    print(f"  Max drawdown: {oos['max_drawdown_pct']:.4f}%")
    print(f"  Calmar: {oos['calmar_ratio']:.4f}")


def main() -> int:
    prices = load_prices()
    rebalances = monthly_rebalances(prices.index, BACKTEST_START)
    if not rebalances:
        raise RuntimeError("no monthly rebalances")
    sim = simulate_spy_gate(prices, rebalances)
    report = build_report(prices, sim)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print_oos_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
