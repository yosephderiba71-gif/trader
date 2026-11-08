"""Read-only backtest for the frozen Trial #25 sector momentum rules."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from momentum_bot.sector_momentum import (
    CASH,
    MARGIN_RATE,
    MAX_SCALAR,
    MIN_SCALAR,
    SECTOR_ASSETS,
    TARGET_VOL,
    TOP_K,
    TRADING_DAYS,
    VOL_WINDOW,
    compute_basket_volatility,
    compute_metrics,
    compute_vol_target_decision,
    volatility_scalar,
)

BENCHMARK = "SPY"
SIM_ASSETS = SECTOR_ASSETS + (CASH,)
SECTOR_SLICE = slice(0, len(SECTOR_ASSETS))
COST_BPS = 10.0
COST_RATE = COST_BPS / 10_000.0
TRAIN_FRACTION = 0.70
BASELINE_RUNS = 1000
BASELINE_PERCENTILE = 95.0
BASELINE_SEED = 20261107
BACKTEST_START = pd.Timestamp("2005-01-01")
TRIAL = 25
RULES_FILE = "research/trial25_vol_target_rules.md"
REPORT_PATH = Path("research/trial25_vol_target_report.json")


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
    symbols = list(SECTOR_ASSETS) + [BENCHMARK]
    prices = pd.concat(
        [download_close(symbol) for symbol in symbols],
        axis=1,
        join="outer",
    ).sort_index()
    prices = prices.loc[:, symbols]
    if len(prices) < 200:
        raise RuntimeError("not enough history")
    return prices


def monthly_rebalances(index: pd.DatetimeIndex, start_date: pd.Timestamp) -> list[int]:
    firsts = index.to_series().groupby(index.to_period("M")).first()
    positions = index.get_indexer(pd.DatetimeIndex(firsts.values))
    return [
        int(position)
        for position in positions
        if int(position) >= 0 and index[int(position)] >= start_date
    ]


def target_array(targets: dict[str, float]) -> np.ndarray:
    return np.array([targets.get(symbol, 0.0) for symbol in SIM_ASSETS], dtype=float)


def build_returns(prices: pd.DataFrame) -> np.ndarray:
    px = prices.loc[:, SECTOR_ASSETS].to_numpy(dtype=float)
    returns = np.full_like(px, np.nan)
    returns[1:] = px[1:] / px[:-1] - 1.0
    cash = np.zeros((len(prices), 1), dtype=float)
    return np.column_stack([returns, cash])


def simulate(
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
    events: list[dict[str, Any]] = []

    def rebalance(index: int, pre: float, current: np.ndarray) -> tuple[float, np.ndarray]:
        decision = compute_vol_target_decision(prices.iloc[: index + 1])
        target = target_array(decision.weights)
        turnover = float(
            np.abs(target[SECTOR_SLICE] - current[SECTOR_SLICE]).sum()
        )
        cost = turnover * COST_RATE
        margin_cost = decision.margin_debit * MARGIN_RATE / 12.0
        post_cost_equity = pre * (1.0 - cost) * (1.0 - margin_cost)
        events.append(
            {
                "index": int(index),
                "date": prices.index[index],
                "pre_cost_equity": float(pre),
                "post_cost_equity": float(post_cost_equity),
                "turnover": float(turnover),
                "cost": float(cost),
                "margin_debit": float(decision.margin_debit),
                "margin_cost": float(margin_cost),
                "asset_orders": int(
                    np.sum(
                        np.abs(target[SECTOR_SLICE] - current[SECTOR_SLICE]) > 1e-12
                    )
                ),
                "scalar": float(decision.scalar),
                "basket_volatility": float(decision.basket_volatility),
                "gross_exposure": float(decision.gross_exposure),
                "selected": list(decision.selected),
            }
        )
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

    weights_history = pd.DataFrame(
        weight_rows,
        index=pd.DatetimeIndex(dates),
        columns=SIM_ASSETS,
    )
    return {
        "equity": pd.Series(values, index=pd.DatetimeIndex(dates)),
        "weights": weights_history,
        "events": events,
    }


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min() * 100.0)


def cagr(start_value: float, end_value: float, start_date: Any, end_date: Any) -> float:
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    if days <= 0 or start_value <= 0 or end_value <= 0:
        return 0.0
    return float(((end_value / start_value) ** (365.25 / days) - 1.0) * 100.0)


def sharpe_ratio(equity: pd.Series) -> float:
    returns = equity.pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        return 0.0
    volatility = float(returns.std(ddof=1))
    if not np.isfinite(volatility) or volatility <= 0.0:
        return 0.0
    return float(returns.mean() / volatility * np.sqrt(TRADING_DAYS))


def calmar_ratio(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = max_drawdown(equity) / 100.0
    if drawdown >= 0.0:
        return 0.0
    annual_return = (
        cagr(
            float(equity.iloc[0]),
            float(equity.iloc[-1]),
            equity.index[0],
            equity.index[-1],
        )
        / 100.0
    )
    return float(annual_return / abs(drawdown))


def round4(value: float) -> float:
    return round(float(value), 4)


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
    wins = sum(1 for record in records if record["return_pct"] > 0.0)
    return {
        "trade_count": int(count),
        "win_rate": float(wins / count) if count else 0.0,
        "asset_order_count": int(sum(record["asset_orders"] for record in records)),
    }


def calendar_year_metrics(
    equity: pd.Series,
    spy: pd.Series,
    year: int,
    oos_start_date: pd.Timestamp,
) -> dict[str, Any]:
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")
    year_equity = equity.loc[(equity.index >= year_start) & (equity.index <= year_end)]
    if year_equity.empty:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "spy_buy_and_hold_return_pct": 0.0,
            "observations": 0,
            "period": "no_data",
        }

    prior_equity = equity.loc[equity.index < year_start]
    if not prior_equity.empty:
        start_value = float(prior_equity.iloc[-1])
        start_date = prior_equity.index[-1]
    else:
        start_value = float(year_equity.iloc[0])
        start_date = year_equity.index[0]
    end_value = float(year_equity.iloc[-1])
    total_return = (end_value / start_value - 1.0) * 100.0

    path = pd.concat(
        [
            pd.Series([start_value], index=pd.DatetimeIndex([start_date])),
            year_equity,
        ]
    )
    drawdown = float((path / path.cummax() - 1.0).min() * 100.0)

    spy_year = spy.loc[(spy.index >= year_start) & (spy.index <= year_end)]
    spy_prior = spy.loc[spy.index < year_start]
    if spy_year.empty:
        spy_return = 0.0
    else:
        spy_start = (
            float(spy_prior.iloc[-1]) if not spy_prior.empty else float(spy_year.iloc[0])
        )
        spy_return = (float(spy_year.iloc[-1]) / spy_start - 1.0) * 100.0

    if year_end < oos_start_date:
        period = "in_sample"
    elif year_start >= oos_start_date:
        period = "out_of_sample"
    else:
        period = "spanning_oos_start"

    return {
        "total_return_pct": round4(total_return),
        "max_drawdown_pct": round4(drawdown),
        "spy_buy_and_hold_return_pct": round4(spy_return),
        "observations": int(len(year_equity)),
        "period": period,
    }


def baseline_context(
    prices: pd.DataFrame,
    oos_rebalances: list[int],
) -> dict[int, tuple[str, ...]]:
    context: dict[int, tuple[str, ...]] = {}
    for index in oos_rebalances:
        metrics = compute_metrics(prices.iloc[: index + 1])
        context[index] = tuple(
            metric.symbol for metric in metrics if metric.eligible
        )
    return context


def random_baseline(
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
    context = baseline_context(prices, oos_rebalances)
    rng = np.random.default_rng(BASELINE_SEED)
    baseline_returns = np.empty(BASELINE_RUNS, dtype=float)
    cash_index = SIM_ASSETS.index(CASH)

    for run in range(BASELINE_RUNS):
        weights = oos_start_weights.copy()
        value = 1.0
        for t in range(oos_start + 1, n):
            r = returns[t]
            r = np.where(np.isfinite(r), r, 0.0)
            port = float(np.dot(weights, r))
            growth = 1.0 + port
            if growth <= 0:
                raise RuntimeError("non-positive random-baseline growth")
            value *= growth
            weights = weights * (1.0 + r) / growth
            if t in oos_set:
                eligible = context[t]
                n_positions = min(TOP_K, len(eligible))
                target = np.zeros(len(SIM_ASSETS), dtype=float)
                if n_positions > 0:
                    chosen = rng.choice(eligible, size=n_positions, replace=False)
                    basket_volatility = compute_basket_volatility(
                        prices.iloc[: t + 1],
                        chosen,
                    )
                    scalar = volatility_scalar(
                        basket_volatility,
                        has_positions=True,
                    )
                    for symbol in chosen:
                        target[SIM_ASSETS.index(symbol)] = scalar / TOP_K
                    target[cash_index] = (
                        1.0 - scalar * n_positions / TOP_K
                    )
                else:
                    target[cash_index] = 1.0
                turnover = float(
                    np.abs(target[SECTOR_SLICE] - weights[SECTOR_SLICE]).sum()
                )
                trade_cost = turnover * COST_RATE
                margin_debit = max(
                    0.0,
                    float(target[SECTOR_SLICE].sum()) - 1.0,
                )
                margin_cost = margin_debit * MARGIN_RATE / 12.0
                value *= (1.0 - trade_cost) * (1.0 - margin_cost)
                weights = target.copy()
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
    }


def build_report(
    prices: pd.DataFrame,
    sim: dict[str, Any],
    rebalances: list[int],
    returns: np.ndarray,
) -> dict[str, Any]:
    equity = sim["equity"]
    weights = sim["weights"]
    events = sim["events"]
    n = len(equity)
    split = max(1, min(int(n * TRAIN_FRACTION), n - 1))
    is_equity = equity.iloc[: split + 1]
    oos_equity = equity.iloc[split:]
    oos_start_date = oos_equity.index[0]
    oos_start = int(prices.index.get_loc(oos_start_date))
    oos_start_weights = weights.loc[oos_start_date].to_numpy(dtype=float)

    is_total = (is_equity.iloc[-1] / is_equity.iloc[0] - 1.0) * 100.0
    oos_total = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) * 100.0
    records = trade_records(events, float(equity.iloc[-1]))
    is_stats = trade_stats([record for record in records if record["index"] < oos_start])
    oos_stats = trade_stats([record for record in records if record["index"] >= oos_start])

    spy = prices[BENCHMARK].dropna()
    spy_oos = float((spy.iloc[-1] / spy.loc[oos_start_date] - 1.0) * 100.0)
    baseline = random_baseline(
        prices,
        returns,
        rebalances,
        oos_start,
        oos_start_weights,
        oos_total,
    )

    finite_vols = [
        float(event["basket_volatility"])
        for event in events
        if np.isfinite(event["basket_volatility"])
    ]
    volatility_targeting = {
        "target_annualized_volatility_pct": round4(TARGET_VOL * 100.0),
        "volatility_window_days": int(VOL_WINDOW),
        "scalar_min": round4(MIN_SCALAR),
        "scalar_max": round4(MAX_SCALAR),
        "margin_rate_pct": round4(MARGIN_RATE * 100.0),
        "average_basket_volatility_pct": (
            round4(float(np.mean(finite_vols)) * 100.0) if finite_vols else 0.0
        ),
        "average_scalar": round4(
            float(np.mean([float(event["scalar"]) for event in events]))
        ),
        "average_gross_exposure": round4(
            float(np.mean([float(event["gross_exposure"]) for event in events]))
        ),
        "max_gross_exposure": round4(
            float(max(float(event["gross_exposure"]) for event in events))
        ),
        "rebalances_with_margin": int(
            sum(1 for event in events if float(event["margin_debit"]) > 0.0)
        ),
    }

    return {
        "strategy": "institutional_cross_sectional_sector_momentum",
        "variant": "volatility_targeting",
        "trial": TRIAL,
        "rules_file": RULES_FILE,
        "universe": list(SECTOR_ASSETS),
        "benchmark": BENCHMARK,
        "cost_bps_per_trade": COST_BPS,
        "rebalance_frequency": "first_trading_day_of_each_month",
        "volatility_targeting": volatility_targeting,
        "data": {
            "source": "yfinance",
            "auto_adjust": True,
            "first_date": str(prices.index[0].date()),
            "last_date": str(prices.index[-1].date()),
            "trading_days": int(len(prices)),
            "sector_observations": {
                symbol: int(prices[symbol].notna().sum()) for symbol in SECTOR_ASSETS
            },
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
            "rebalance_count_is": int(sum(1 for event in events if event["index"] < oos_start)),
            "rebalance_count_oos": int(
                sum(1 for event in events if event["index"] >= oos_start)
            ),
        },
        "in_sample": {
            "raw_return_pct": round4(is_total),
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
            **is_stats,
        },
        "out_of_sample": {
            "raw_return_pct": round4(oos_total),
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
        "calendar_year_metrics": {
            str(year): calendar_year_metrics(equity, spy, year, oos_start_date)
            for year in (2008, 2022)
        },
        "definitions": {
            "universe": "All 11 SPDR sectors; a sector without at least 200 valid observations is ineligible.",
            "signal_timing": "Signals use closes through the rebalance date; new weights earn the next day's return. No lookahead.",
            "skip_month_return": "(Price[t-20] / Price[t-120]) - 1.",
            "cash": "Unallocated weight earns 0%; negative cash represents a margin debit.",
            "volatility_target": "At each monthly rebalance, compute the last 60 daily simple returns of the equal-weight selected-sector basket (1/3 per selected sector; any unallocated remainder is cash with zero return), annualize the sample standard deviation by sqrt(252), set scalar = 0.15 / basket volatility, clip the scalar to [0.50, 1.35], and multiply each selected-sector weight by the scalar. Cash is 1 - gross exposure and may be negative.",
            "margin_cost": "When gross exposure exceeds 1.0, a 5% annual rate is charged monthly on the margin debit: (gross exposure - 1) * 5% / 12.",
            "cost_model": "10bp applied to one-way sector-weight turnover; cash and margin-debit changes are excluded from trade turnover and charged through the margin cost.",
            "trade_count": "Rebalance dates with non-zero sector turnover.",
            "win_rate": "Share of counted rebalance trades with a positive return over the following holding period.",
            "max_drawdown_pct": "Peak-to-trough drawdown as a negative percentage.",
            "cagr_pct": "Calendar-day compound annual growth rate.",
            "sharpe_ratio": "Annualized daily Sharpe ratio with a 0% risk-free rate; daily returns use sample standard deviation (ddof=1).",
            "calmar_ratio": "CAGR divided by the absolute max drawdown.",
            "random_baseline": "1000 random monthly schedules. At each OOS rebalance, the baseline samples the same number of positions as the strategy from the same eligible set, without replacement; selected sectors get 1/3 each times the same volatility scalar computed from the baseline's own 60-day equal-weight basket, cash is the residual, and the same 10bp trade and 5% annual margin costs apply. It starts from the strategy's OOS-start weights and uses the same dates.",
            "calendar_year_metrics": "Calendar-year total return and intra-year peak-to-trough drawdown; 2008 is in-sample, 2022 is out-of-sample under the 70/30 split.",
        },
    }


def print_oos_summary(report: dict[str, Any]) -> None:
    oos = report["out_of_sample"]
    print("\nTrial #25 OOS summary:")
    print(f"  Raw return: {oos['raw_return_pct']:.4f}%")
    print(f"  Max drawdown: {oos['max_drawdown_pct']:.4f}%")
    print(f"  Sharpe: {oos['sharpe_ratio']:.4f}")
    print(f"  Calmar: {oos['calmar_ratio']:.4f}")


def main() -> int:
    prices = load_prices()
    rebalances = monthly_rebalances(prices.index, BACKTEST_START)
    if not rebalances:
        raise RuntimeError("no monthly rebalances")
    returns = build_returns(prices)
    sim = simulate(prices, rebalances, returns)
    report = build_report(prices, sim, rebalances, returns)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print_oos_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
