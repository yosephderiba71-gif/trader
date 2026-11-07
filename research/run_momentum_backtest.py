"""Read-only backtest for the frozen multi-asset time-series momentum rules."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from momentum_bot.signals import ASSETS, SMA_WINDOW, compute_target_weights

COST_BPS = 10.0
COST_RATE = COST_BPS / 10_000.0
TRAIN_FRACTION = 0.70
BASELINE_RUNS = 1000
BASELINE_PERCENTILE = 95.0
BASELINE_SEED = 20261107
REPORT_PATH = Path("research/momentum_backtest_report.json")


def download_close(symbol: str) -> pd.Series:
    import yfinance as yf

    err = None
    for attempt in range(3):
        try:
            frame = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
            if frame is not None and not frame.empty and "Close" in frame.columns:
                close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
                if not close.empty:
                    idx = pd.DatetimeIndex(close.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    close.index = idx.normalize()
                    return close[~close.index.duplicated(keep="last")].sort_index().rename(symbol)
            err = f"empty data for {symbol}"
        except Exception as exc:
            err = str(exc)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {symbol}: {err}")


def load_prices() -> pd.DataFrame:
    prices = pd.concat([download_close(s) for s in ASSETS], axis=1, join="inner").dropna()
    prices = prices.loc[:, list(ASSETS)].sort_index()
    if len(prices) < SMA_WINDOW + 2:
        raise RuntimeError("not enough common history")
    return prices


def monthly_rebalances(index: pd.DatetimeIndex, min_index: int) -> list[int]:
    firsts = index.to_series().groupby(index.to_period("M")).first()
    positions = index.get_indexer(pd.DatetimeIndex(firsts.values))
    return [int(p) for p in positions if int(p) >= min_index]


def target_array(targets: dict[str, float]) -> np.ndarray:
    return np.array([targets.get(s, 0.0) for s in ASSETS], dtype=float)


def simulate(prices: pd.DataFrame, rebalances: list[int]) -> dict[str, Any]:
    n = len(prices)
    px = prices.to_numpy(dtype=float)
    rets = np.zeros_like(px)
    rets[1:] = px[1:] / px[:-1] - 1.0
    rebalance_set = set(rebalances)
    first = rebalances[0]
    weights = np.zeros(len(ASSETS))
    equity = 1.0
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    events: list[dict[str, Any]] = []

    def rebalance(index: int, pre: float, current: np.ndarray) -> tuple[float, np.ndarray]:
        targets = compute_target_weights(prices.iloc[: index + 1])
        target = target_array(targets)
        turnover = float(np.abs(target - current).sum())
        cost = turnover * COST_RATE
        events.append(
            {
                "index": int(index),
                "date": prices.index[index],
                "pre_cost_equity": float(pre),
                "turnover": float(turnover),
                "cost": float(cost),
                "asset_orders": int(np.sum(np.abs(target - current) > 1e-12)),
            }
        )
        return pre * (1.0 - cost), target

    equity, weights = rebalance(first, equity, weights)
    dates.append(prices.index[first])
    values.append(float(equity))

    for t in range(first + 1, n):
        r = rets[t]
        port = float(np.dot(weights, r))
        growth = 1.0 + port
        if growth <= 0:
            raise RuntimeError(f"non-positive equity growth on {prices.index[t]}")
        equity *= growth
        weights = weights * (1.0 + r) / growth
        if t in rebalance_set:
            equity, weights = rebalance(t, equity, weights)
        dates.append(prices.index[t])
        values.append(float(equity))

    return {"equity": pd.Series(values, index=pd.DatetimeIndex(dates)), "events": events}


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min() * 100.0)


def cagr(start_value: float, end_value: float, start_date: Any, end_date: Any) -> float:
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    if days <= 0 or start_value <= 0 or end_value <= 0:
        return 0.0
    return float(((end_value / start_value) ** (365.25 / days) - 1.0) * 100.0)


def trade_records(events: list[dict[str, Any]], final_equity: float) -> list[dict[str, Any]]:
    records = []
    for i, event in enumerate(events):
        if event["turnover"] <= 1e-12:
            continue
        end_equity = events[i + 1]["pre_cost_equity"] if i + 1 < len(events) else final_equity
        records.append(
            {
                "index": event["index"],
                "return_pct": float((end_equity / event["pre_cost_equity"] - 1.0) * 100.0),
                "asset_orders": int(event["asset_orders"]),
            }
        )
    return records


def trade_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    wins = sum(1 for r in records if r["return_pct"] > 0.0)
    return {
        "trade_count": int(count),
        "win_rate": float(wins / count) if count else 0.0,
        "asset_order_count": int(sum(r["asset_orders"] for r in records)),
    }


def random_baseline(
    prices: pd.DataFrame, oos_start: int, rebalances: list[int], strategy_return: float
) -> dict[str, Any]:
    n = len(prices)
    px = prices.to_numpy(dtype=float)
    rets = np.zeros_like(px)
    rets[1:] = px[1:] / px[:-1] - 1.0
    oos_rebalances = [i for i in rebalances if i > oos_start]
    oos_set = set(oos_rebalances)
    charge_initial = oos_start in set(rebalances)
    n_events = 1 + len(oos_rebalances)
    rng = np.random.default_rng(BASELINE_SEED)
    draws = rng.dirichlet(np.ones(len(ASSETS) + 1), size=(BASELINE_RUNS, n_events))
    returns = np.empty(BASELINE_RUNS, dtype=float)
    for run in range(BASELINE_RUNS):
        weights = draws[run, 0, : len(ASSETS)].copy()
        value = 1.0 - (float(weights.sum()) * COST_RATE if charge_initial else 0.0)
        event = 1
        for t in range(oos_start + 1, n):
            r = rets[t]
            port = float(np.dot(weights, r))
            growth = 1.0 + port
            if growth <= 0:
                raise RuntimeError("non-positive random-baseline growth")
            value *= growth
            weights = weights * (1.0 + r) / growth
            if t in oos_set:
                target = draws[run, event, : len(ASSETS)]
                value *= 1.0 - float(np.abs(target - weights).sum()) * COST_RATE
                weights = target.copy()
                event += 1
        returns[run] = (value - 1.0) * 100.0
    p95 = float(np.percentile(returns, BASELINE_PERCENTILE))
    return {
        "runs": BASELINE_RUNS,
        "seed": BASELINE_SEED,
        "p95_return_pct": p95,
        "mean_return_pct": float(returns.mean()),
        "median_return_pct": float(np.median(returns)),
        "p05_return_pct": float(np.percentile(returns, 5.0)),
        "std_return_pct": float(returns.std(ddof=1)),
        "strategy_percentile": float((returns <= strategy_return).mean() * 100.0),
    }


def round4(value: float) -> float:
    return round(float(value), 4)


def build_report(prices: pd.DataFrame, sim: dict[str, Any], rebalances: list[int]) -> dict[str, Any]:
    equity = sim["equity"]
    events = sim["events"]
    n = len(equity)
    split = max(1, min(int(n * TRAIN_FRACTION), n - 1))
    is_eq = equity.iloc[: split + 1]
    oos_eq = equity.iloc[split:]
    oos_start_date = oos_eq.index[0]
    oos_start = int(prices.index.get_loc(oos_start_date))

    is_total = (is_eq.iloc[-1] / is_eq.iloc[0] - 1.0) * 100.0
    oos_total = (oos_eq.iloc[-1] / oos_eq.iloc[0] - 1.0) * 100.0
    records = trade_records(events, float(equity.iloc[-1]))
    is_stats = trade_stats([r for r in records if r["index"] < oos_start])
    oos_stats = trade_stats([r for r in records if r["index"] >= oos_start])

    spy = prices["SPY"]
    spy_oos = float((spy.iloc[-1] / spy.loc[oos_start_date] - 1.0) * 100.0)
    baseline = random_baseline(prices, oos_start, rebalances, oos_total)

    return {
        "strategy": "multi_asset_time_series_momentum",
        "rules_file": "research/momentum_rules.md",
        "universe": list(ASSETS),
        "cost_bps_per_trade": COST_BPS,
        "rebalance_frequency": "first_trading_day_of_each_month",
        "data": {
            "source": "yfinance",
            "auto_adjust": True,
            "first_date": str(prices.index[0].date()),
            "last_date": str(prices.index[-1].date()),
            "common_trading_days": int(len(prices)),
            "backtest_start": str(equity.index[0].date()),
            "backtest_end": str(equity.index[-1].date()),
            "backtest_observations": int(n),
            "is_start": str(is_eq.index[0].date()),
            "is_end": str(is_eq.index[-1].date()),
            "is_daily_returns": int(split),
            "oos_start": str(oos_eq.index[0].date()),
            "oos_end": str(oos_eq.index[-1].date()),
            "oos_daily_returns": int(n - 1 - split),
            "rebalance_count_total": int(len(events)),
            "rebalance_count_is": int(sum(1 for e in events if e["index"] < oos_start)),
            "rebalance_count_oos": int(sum(1 for e in events if e["index"] >= oos_start)),
        },
        "in_sample": {
            "total_return_pct": round4(is_total),
            "cagr_pct": round4(cagr(is_eq.iloc[0], is_eq.iloc[-1], is_eq.index[0], is_eq.index[-1])),
            "max_drawdown_pct": round4(max_drawdown(is_eq)),
            **is_stats,
        },
        "out_of_sample": {
            "total_return_pct": round4(oos_total),
            "cagr_pct": round4(cagr(oos_eq.iloc[0], oos_eq.iloc[-1], oos_eq.index[0], oos_eq.index[-1])),
            "max_drawdown_pct": round4(max_drawdown(oos_eq)),
            **oos_stats,
        },
        "buy_and_hold_spy_oos_return_pct": round4(spy_oos),
        "random_baseline": {
            "runs": baseline["runs"],
            "seed": baseline["seed"],
            "p95_return_pct": round4(baseline["p95_return_pct"]),
            "mean_return_pct": round4(baseline["mean_return_pct"]),
            "median_return_pct": round4(baseline["median_return_pct"]),
            "p05_return_pct": round4(baseline["p05_return_pct"]),
            "std_return_pct": round4(baseline["std_return_pct"]),
            "strategy_percentile": round4(baseline["strategy_percentile"]),
        },
        "strategy_beat_random_baseline": bool(oos_total > baseline["p95_return_pct"]),
        "definitions": {
            "trade_count": "OOS monthly rebalance trades with non-zero turnover, counted by rebalance date.",
            "win_rate": "Share of counted monthly rebalance trades with a positive return over the following holding period.",
            "max_drawdown_pct": "Peak-to-trough drawdown as a negative percentage.",
            "cagr_pct": "Calendar-day compound annual growth rate.",
            "cost_model": "10bp applied to one-way turnover (sum of absolute asset-weight changes).",
            "signal_timing": "Signals use closes through the rebalance date; new weights earn the next day's return. No lookahead.",
            "random_baseline": "1000 random monthly weight schedules drawn from Dirichlet(1,1,1,1) over SPY/TLT/GLD/cash on the same OOS dates and cost model.",
        },
    }


def main() -> int:
    prices = load_prices()
    rebalances = monthly_rebalances(prices.index, SMA_WINDOW - 1)
    sim = simulate(prices, rebalances)
    report = build_report(prices, sim, rebalances)
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
