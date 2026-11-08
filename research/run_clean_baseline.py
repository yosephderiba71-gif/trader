"""Clean, unlevered random baseline for Trial #24.

This is a read-only audit script. It deliberately does not modify the Trial #24
or Trial #25 runners. The clean baseline samples three sectors from the full
11-sector universe at each OOS rebalance, holds equal 1/3 weights, and applies
no 200-SMA eligibility filter, no volatility targeting, and no margin.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from momentum_bot.sector_momentum import (
    CASH,
    SECTOR_ASSETS,
    TOP_K,
    compute_target_weights,
)
from research.run_sector_momentum import (
    BACKTEST_START,
    BASELINE_PERCENTILE,
    BASELINE_RUNS,
    BASELINE_SEED,
    BENCHMARK,
    COST_RATE,
    SIM_ASSETS,
    SECTOR_SLICE,
    TRAIN_FRACTION,
    build_returns,
    cagr,
    load_prices,
    max_drawdown,
    monthly_rebalances,
    round4,
)

REPORT_PATH = Path("research/clean_baseline_report.json")


def _target_array(targets: dict[str, float]) -> np.ndarray:
    return np.array([targets.get(symbol, 0.0) for symbol in SIM_ASSETS], dtype=float)


def simulate_trial24(
    prices: pd.DataFrame,
    rebalances: list[int],
    returns: np.ndarray,
) -> dict[str, Any]:
    n = len(prices)
    rebalance_set = set(rebalances)
    first = rebalances[0]
    weights = np.zeros(len(SIM_ASSETS), dtype=float)
    equity = 1.0
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    weight_rows: list[np.ndarray] = []

    def rebalance(index: int, pre: float, current: np.ndarray) -> tuple[float, np.ndarray]:
        target = _target_array(compute_target_weights(prices.iloc[: index + 1]))
        turnover = float(np.abs(target[SECTOR_SLICE] - current[SECTOR_SLICE]).sum())
        post_cost_equity = pre * (1.0 - turnover * COST_RATE)
        return post_cost_equity, target

    equity, weights = rebalance(first, equity, weights)
    dates.append(prices.index[first])
    values.append(float(equity))
    weight_rows.append(weights.copy())

    for t in range(first + 1, n):
        r = returns[t]
        r = np.where(np.isfinite(r), r, 0.0)
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
        weight_rows.append(weights.copy())

    return {
        "equity": pd.Series(values, index=pd.DatetimeIndex(dates)),
        "weights": pd.DataFrame(
            weight_rows,
            index=pd.DatetimeIndex(dates),
            columns=SIM_ASSETS,
        ),
    }


def clean_random_baseline(
    prices: pd.DataFrame,
    returns: np.ndarray,
    rebalances: list[int],
    oos_start: int,
    oos_start_weights: np.ndarray,
    strategy_return: float,
) -> dict[str, Any]:
    n = len(prices)
    oos_rebalances = [index for index in rebalances if index > oos_start]
    oos_set = set(oos_rebalances)
    rng = np.random.default_rng(BASELINE_SEED)
    baseline_returns = np.empty(BASELINE_RUNS, dtype=float)
    cash_index = SIM_ASSETS.index(CASH)
    universe = np.array(SECTOR_ASSETS)

    for run in range(BASELINE_RUNS):
        weights = oos_start_weights.copy()
        value = 1.0
        for t in range(oos_start + 1, n):
            r = returns[t]
            r = np.where(np.isfinite(r), r, 0.0)
            port = float(np.dot(weights, r))
            growth = 1.0 + port
            if growth <= 0:
                raise RuntimeError("non-positive clean-baseline growth")
            value *= growth
            weights = weights * (1.0 + r) / growth
            if t in oos_set:
                chosen = rng.choice(universe, size=TOP_K, replace=False)
                target = np.zeros(len(SIM_ASSETS), dtype=float)
                for symbol in chosen:
                    target[SIM_ASSETS.index(str(symbol))] = 1.0 / TOP_K
                target[cash_index] = 0.0
                turnover = float(
                    np.abs(target[SECTOR_SLICE] - weights[SECTOR_SLICE]).sum()
                )
                value *= 1.0 - turnover * COST_RATE
                weights = target
        baseline_returns[run] = (value - 1.0) * 100.0

    p95 = float(np.percentile(baseline_returns, BASELINE_PERCENTILE))
    return {
        "runs": BASELINE_RUNS,
        "seed": BASELINE_SEED,
        "p95_return_pct": p95,
        "mean_return_pct": float(baseline_returns.mean()),
        "median_return_pct": float(np.median(baseline_returns)),
        "p05_return_pct": float(np.percentile(baseline_returns, 5.0)),
        "std_return_pct": float(baseline_returns.std(ddof=1)),
        "strategy_percentile": float((baseline_returns <= strategy_return).mean() * 100.0),
        "oos_rebalances": len(oos_rebalances),
    }


def main() -> int:
    prices = load_prices()
    rebalances = monthly_rebalances(prices.index, BACKTEST_START)
    returns = build_returns(prices)
    sim = simulate_trial24(prices, rebalances, returns)
    equity = sim["equity"]
    weights = sim["weights"]
    n = len(equity)
    split = max(1, min(int(n * TRAIN_FRACTION), n - 1))
    is_equity = equity.iloc[: split + 1]
    oos_equity = equity.iloc[split:]
    oos_start_date = oos_equity.index[0]
    oos_start = int(prices.index.get_loc(oos_start_date))
    oos_start_weights = weights.loc[oos_start_date].to_numpy(dtype=float)
    oos_total = float((oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) * 100.0)
    baseline = clean_random_baseline(
        prices,
        returns,
        rebalances,
        oos_start,
        oos_start_weights,
        oos_total,
    )
    report = {
        "strategy": "institutional_cross_sectional_sector_momentum",
        "trial": 24,
        "variant": "clean_unlevered_random_baseline",
        "universe": list(SECTOR_ASSETS),
        "benchmark": BENCHMARK,
        "cost_bps_per_trade": COST_RATE * 10_000.0,
        "baseline_definition": (
            "1000 random monthly schedules. At each OOS rebalance, sample 3 sectors "
            "from the full 11-sector universe without replacement, equal 1/3 each, "
            "no 200-SMA eligibility filter, no volatility targeting, no margin. "
            "Starts from the Trial #24 OOS-start weights and uses the same dates "
            "and 10bp one-way turnover cost."
        ),
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
            "rebalance_count_total": int(len(rebalances)),
            "rebalance_count_oos": int(baseline["oos_rebalances"]),
        },
        "trial24_out_of_sample": {
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
        },
        "clean_random_baseline": {
            "runs": baseline["runs"],
            "seed": baseline["seed"],
            "p95_return_pct": round4(baseline["p95_return_pct"]),
            "mean_return_pct": round4(baseline["mean_return_pct"]),
            "median_return_pct": round4(baseline["median_return_pct"]),
            "p05_return_pct": round4(baseline["p05_return_pct"]),
            "std_return_pct": round4(baseline["std_return_pct"]),
            "strategy_percentile": round4(baseline["strategy_percentile"]),
        },
        "strategy_beat_clean_baseline": bool(oos_total > baseline["p95_return_pct"]),
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
