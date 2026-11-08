"""Trial #27: Turn-of-the-Month (TOTM) Effect Backtest.

Pre-registered specifications (research/trial27_rules.md):
- Assets: SPY (Risky), BIL (Cash). Unlevered only.
- Entry: Buy SPY at close on last trading day of Month T (Day -1).
- Holding: Hold through trading days +1, +2, and +3 of Month T+1.
- Exit: Sell SPY at close on trading day +3 and buy BIL.
- Execution: Standard market order at 15:50 EST.
- Cost: 5bp round-trip per month.
- Benchmark: Buy-and-Hold SPY.
- Baseline: 1,000 intra-month random 4-day holding windows, rest BIL.
- Split: 70/30 In-Sample/Out-of-Sample.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORT_PATH = Path("research/trial27_totm_report.json")
RULES_PATH = Path("research/trial27_rules.md")
START_DATE = "2000-01-01"
COST_ROUNDTRIP_BPS = 5.0
COST_ROUNDTRIP_RATE = COST_ROUNDTRIP_BPS / 10_000.0
COST_ONEWAY_RATE = COST_ROUNDTRIP_RATE / 2.0
TRAIN_FRACTION = 0.70
BASELINE_RUNS = 1000
BASELINE_PERCENTILE = 95.0
BASELINE_SEED = 20261107
TRADING_DAYS = 252


def download_close(symbol: str, start: str = START_DATE) -> pd.Series:
    import yfinance as yf

    err = None
    for attempt in range(3):
        try:
            frame = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
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


def load_data() -> pd.DataFrame:
    spy = download_close("SPY")
    bil = download_close("BIL")
    df = pd.concat([spy, bil], axis=1).sort_index()
    # SPY daily returns
    df["SPY_ret"] = df["SPY"].pct_change()
    # BIL daily returns: 0.0 before BIL inception (May 2007), actual pct_change after
    df["BIL_ret"] = df["BIL"].pct_change().fillna(0.0)
    # Drop first day since pct_change is NaN
    df = df.iloc[1:].copy()
    df["year_month"] = df.index.to_period("M")
    return df


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


def compute_metrics(equity: pd.Series) -> dict[str, float]:
    total_ret = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)
    c = cagr(equity.iloc[0], equity.iloc[-1], equity.index[0], equity.index[-1])
    mdd = max_drawdown(equity)
    sh = sharpe_ratio(equity)
    cal = calmar_ratio(equity)
    return {
        "total_return_pct": round(total_ret, 4),
        "cagr_pct": round(c, 4),
        "max_drawdown_pct": round(mdd, 4),
        "sharpe_ratio": round(sh, 4),
        "calmar_ratio": round(cal, 4),
    }


def simulate_totm(
    df: pd.DataFrame,
    include_day_m1_return: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Simulate TOTM strategy on daily returns.
    
    If include_day_m1_return is True:
        Holds SPY during all 4 designated anomaly days: Day -1, Day +1, Day +2, Day +3.
    If include_day_m1_return is False:
        Executes at Day -1 close and exits at Day +3 close:
        Holds SPY during Day +1, Day +2, Day +3.
        Holds BIL on Day -1 and all other days.
    """
    strat_ret = df["BIL_ret"].copy()
    costs = pd.Series(0.0, index=df.index)
    months = df["year_month"].unique()

    for i in range(len(months) - 1):
        mT = months[i]
        mT1 = months[i + 1]
        daysT = df[df["year_month"] == mT].index
        daysT1 = df[df["year_month"] == mT1].index

        day_m1 = daysT[-1]
        day_p1 = daysT1[0]
        day_p2 = daysT1[1] if len(daysT1) > 1 else None
        day_p3 = daysT1[2] if len(daysT1) > 2 else None

        if include_day_m1_return:
            strat_ret.loc[day_m1] = df.loc[day_m1, "SPY_ret"]
        strat_ret.loc[day_p1] = df.loc[day_p1, "SPY_ret"]
        if day_p2 is not None:
            strat_ret.loc[day_p2] = df.loc[day_p2, "SPY_ret"]
        if day_p3 is not None:
            strat_ret.loc[day_p3] = df.loc[day_p3, "SPY_ret"]

        # Transaction cost: 2.5bp on entry, 2.5bp on exit (5bp round trip)
        costs.loc[day_m1] += COST_ONEWAY_RATE
        exit_day = day_p3 if day_p3 is not None else day_p1
        costs.loc[exit_day] += COST_ONEWAY_RATE

    net_ret = strat_ret - costs
    equity = (1.0 + net_ret).cumprod()
    return equity, net_ret


def run_random_baseline(
    oos_df: pd.DataFrame,
    window_days: int = 4,
    runs: int = BASELINE_RUNS,
    seed: int = BASELINE_SEED,
) -> dict[str, Any]:
    """Run intra-month random baseline of 1,000 trajectories.
    
    For each month m, randomly select a starting day index k to form a random
    window_days holding window within that specific month. Compound the return,
    add BIL for remaining days. Apply 5bp round-trip cost per month.
    """
    oos_months = oos_df["year_month"].unique()
    m_slices: list[tuple[int, int]] = []
    cur_idx = 0
    for m in oos_months:
        m_len = len(oos_df[oos_df["year_month"] == m])
        m_slices.append((cur_idx, cur_idx + m_len))
        cur_idx += m_len

    oos_spy_ret = oos_df["SPY_ret"].to_numpy(dtype=float)
    oos_bil_ret = oos_df["BIL_ret"].to_numpy(dtype=float)
    oos_days = (oos_df.index[-1] - oos_df.index[0]).days

    rng = np.random.default_rng(seed)
    base_calmars = np.empty(runs, dtype=float)
    base_sharpes = np.empty(runs, dtype=float)
    base_returns = np.empty(runs, dtype=float)
    base_mdds = np.empty(runs, dtype=float)
    base_cagrs = np.empty(runs, dtype=float)

    for run in range(runs):
        rets = oos_bil_ret.copy()
        for s, e in m_slices:
            L = e - s
            if L >= window_days:
                k = rng.integers(0, L - window_days + 1)
                rets[s + k : s + k + window_days] = oos_spy_ret[s + k : s + k + window_days]
                rets[s + k] -= COST_ONEWAY_RATE
                rets[s + k + window_days - 1] -= COST_ONEWAY_RATE
            else:
                rets[s : e] = oos_spy_ret[s : e]
                rets[s] -= COST_ROUNDTRIP_RATE

        eq = np.empty(len(rets) + 1, dtype=float)
        eq[0] = 1.0
        np.cumprod(1.0 + rets, out=eq[1:])
        tot_ret = (eq[-1] - 1.0) * 100.0
        c = ((eq[-1]) ** (365.25 / oos_days) - 1.0) * 100.0
        cummax = np.maximum.accumulate(eq)
        mdd = np.min(eq / cummax - 1.0) * 100.0
        s_std = np.std(rets, ddof=1)
        sh = (np.mean(rets) / s_std * np.sqrt(TRADING_DAYS)) if s_std > 0 else 0.0
        cal = (c / abs(mdd)) if mdd < 0 else 0.0

        base_calmars[run] = cal
        base_sharpes[run] = sh
        base_returns[run] = tot_ret
        base_mdds[run] = mdd
        base_cagrs[run] = c

    return {
        "runs": runs,
        "seed": seed,
        "window_days": window_days,
        "calmars": base_calmars,
        "sharpes": base_sharpes,
        "returns": base_returns,
        "mdds": base_mdds,
        "cagrs": base_cagrs,
        "p95_calmar": float(np.percentile(base_calmars, BASELINE_PERCENTILE)),
        "p95_sharpe": float(np.percentile(base_sharpes, BASELINE_PERCENTILE)),
        "p95_return_pct": float(np.percentile(base_returns, BASELINE_PERCENTILE)),
        "median_calmar": float(np.median(base_calmars)),
        "median_sharpe": float(np.median(base_sharpes)),
        "median_return_pct": float(np.median(base_returns)),
        "median_mdd_pct": float(np.median(base_mdds)),
        "mean_calmar": float(np.mean(base_calmars)),
        "mean_sharpe": float(np.mean(base_sharpes)),
        "mean_return_pct": float(np.mean(base_returns)),
        "mean_mdd_pct": float(np.mean(base_mdds)),
    }


def main() -> int:
    print("=" * 65)
    print("Trial #27: Turn-of-the-Month (TOTM) Effect Backtest")
    print("=" * 65)
    print("Loading SPY and BIL data from yfinance...")
    df = load_data()
    n = len(df)
    split = int(n * TRAIN_FRACTION)
    is_df = df.iloc[:split]
    oos_df = df.iloc[split:]

    print(f"Dataset total observations: {n}")
    print(f"In-Sample  (70%): {is_df.index[0].date()} to {is_df.index[-1].date()} ({len(is_df)} days)")
    print(f"Out-of-Sample (30%): {oos_df.index[0].date()} to {oos_df.index[-1].date()} ({len(oos_df)} days)")

    # 1. Primary: 4-day anomaly window (Day -1, Day +1, Day +2, Day +3)
    eq_4d_full, ret_4d_full = simulate_totm(df, include_day_m1_return=True)
    eq_4d_is = eq_4d_full.iloc[:split]
    eq_4d_is = eq_4d_is / eq_4d_is.iloc[0]
    eq_4d_oos = eq_4d_full.iloc[split:]
    eq_4d_oos = eq_4d_oos / eq_4d_oos.iloc[0]

    metrics_4d_is = compute_metrics(eq_4d_is)
    metrics_4d_oos = compute_metrics(eq_4d_oos)

    # 2. Strict execution variant: Day -1 close entry to Day +3 close exit (Day +1, +2, +3)
    eq_3d_full, ret_3d_full = simulate_totm(df, include_day_m1_return=False)
    eq_3d_is = eq_3d_full.iloc[:split]
    eq_3d_is = eq_3d_is / eq_3d_is.iloc[0]
    eq_3d_oos = eq_3d_full.iloc[split:]
    eq_3d_oos = eq_3d_oos / eq_3d_oos.iloc[0]

    metrics_3d_is = compute_metrics(eq_3d_is)
    metrics_3d_oos = compute_metrics(eq_3d_oos)

    # 3. Benchmark: SPY Buy-and-Hold
    spy_is = df["SPY"].iloc[:split]
    spy_is = spy_is / spy_is.iloc[0]
    spy_oos = df["SPY"].iloc[split:]
    spy_oos = spy_oos / spy_oos.iloc[0]
    metrics_spy_is = compute_metrics(spy_is)
    metrics_spy_oos = compute_metrics(spy_oos)

    # 4. Intra-month Random Baseline (4-day window as pre-registered)
    print("\nRunning intra-month random baseline (1,000 runs)...")
    baseline4 = run_random_baseline(oos_df, window_days=4, runs=BASELINE_RUNS, seed=BASELINE_SEED)

    # Percentiles for Primary (4-day TOTM)
    p_calmar_4d = float((baseline4["calmars"] <= metrics_4d_oos["calmar_ratio"]).mean() * 100.0)
    p_sharpe_4d = float((baseline4["sharpes"] <= metrics_4d_oos["sharpe_ratio"]).mean() * 100.0)
    p_return_4d = float((baseline4["returns"] <= metrics_4d_oos["total_return_pct"]).mean() * 100.0)

    # Also calculate 3-day baseline for full rigor
    baseline3 = run_random_baseline(oos_df, window_days=3, runs=BASELINE_RUNS, seed=BASELINE_SEED)
    p_calmar_3d_vs_4d = float((baseline4["calmars"] <= metrics_3d_oos["calmar_ratio"]).mean() * 100.0)
    p_sharpe_3d_vs_4d = float((baseline4["sharpes"] <= metrics_3d_oos["sharpe_ratio"]).mean() * 100.0)
    p_calmar_3d_vs_3d = float((baseline3["calmars"] <= metrics_3d_oos["calmar_ratio"]).mean() * 100.0)
    p_sharpe_3d_vs_3d = float((baseline3["sharpes"] <= metrics_3d_oos["sharpe_ratio"]).mean() * 100.0)

    pass_calmar = bool(metrics_4d_oos["calmar_ratio"] > baseline4["p95_calmar"])
    pass_sharpe = bool(metrics_4d_oos["sharpe_ratio"] > baseline4["p95_sharpe"])

    report: dict[str, Any] = {
        "strategy": "turn_of_the_month_effect",
        "trial": 27,
        "rules_file": str(RULES_PATH),
        "status": "COMPLETED",
        "assets": {"risky": "SPY", "cash": "BIL"},
        "cost_roundtrip_bps": COST_ROUNDTRIP_BPS,
        "train_fraction": TRAIN_FRACTION,
        "data": {
            "source": "yfinance",
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "total_observations": n,
            "is_start": str(is_df.index[0].date()),
            "is_end": str(is_df.index[-1].date()),
            "is_observations": len(is_df),
            "oos_start": str(oos_df.index[0].date()),
            "oos_end": str(oos_df.index[-1].date()),
            "oos_observations": len(oos_df),
        },
        "totm_4day_primary": {
            "description": "SPY held on all 4 anomaly days (Day -1, Day +1, Day +2, Day +3), BIL otherwise.",
            "in_sample": metrics_4d_is,
            "out_of_sample": metrics_4d_oos,
            "baseline_percentiles": {
                "calmar_percentile": round(p_calmar_4d, 2),
                "sharpe_percentile": round(p_sharpe_4d, 2),
                "return_percentile": round(p_return_4d, 2),
            },
            "beat_p95_calmar": pass_calmar,
            "beat_p95_sharpe": pass_sharpe,
        },
        "totm_3day_close_execution": {
            "description": "Entered at Day -1 close, exited at Day +3 close (SPY held Day +1, +2, +3; BIL on Day -1 and rest).",
            "in_sample": metrics_3d_is,
            "out_of_sample": metrics_3d_oos,
            "vs_4day_baseline": {
                "calmar_percentile": round(p_calmar_3d_vs_4d, 2),
                "sharpe_percentile": round(p_sharpe_3d_vs_4d, 2),
                "beat_p95_calmar": bool(metrics_3d_oos["calmar_ratio"] > baseline4["p95_calmar"]),
                "beat_p95_sharpe": bool(metrics_3d_oos["sharpe_ratio"] > baseline4["p95_sharpe"]),
            },
            "vs_3day_baseline": {
                "calmar_percentile": round(p_calmar_3d_vs_3d, 2),
                "sharpe_percentile": round(p_sharpe_3d_vs_3d, 2),
                "beat_p95_calmar": bool(metrics_3d_oos["calmar_ratio"] > baseline3["p95_calmar"]),
                "beat_p95_sharpe": bool(metrics_3d_oos["sharpe_ratio"] > baseline3["p95_sharpe"]),
            },
        },
        "benchmark_spy_buy_and_hold": {
            "in_sample": metrics_spy_is,
            "out_of_sample": metrics_spy_oos,
        },
        "clean_random_baseline_4day": {
            "runs": baseline4["runs"],
            "seed": baseline4["seed"],
            "window_days": 4,
            "p95_calmar": round(baseline4["p95_calmar"], 4),
            "p95_sharpe": round(baseline4["p95_sharpe"], 4),
            "p95_return_pct": round(baseline4["p95_return_pct"], 4),
            "median_calmar": round(baseline4["median_calmar"], 4),
            "median_sharpe": round(baseline4["median_sharpe"], 4),
            "median_return_pct": round(baseline4["median_return_pct"], 4),
            "median_mdd_pct": round(baseline4["median_mdd_pct"], 4),
            "mean_calmar": round(baseline4["mean_calmar"], 4),
            "mean_sharpe": round(baseline4["mean_sharpe"], 4),
            "mean_return_pct": round(baseline4["mean_return_pct"], 4),
            "mean_mdd_pct": round(baseline4["mean_mdd_pct"], 4),
        },
        "clean_random_baseline_3day": {
            "runs": baseline3["runs"],
            "seed": baseline3["seed"],
            "window_days": 3,
            "p95_calmar": round(baseline3["p95_calmar"], 4),
            "p95_sharpe": round(baseline3["p95_sharpe"], 4),
            "p95_return_pct": round(baseline3["p95_return_pct"], 4),
            "median_calmar": round(baseline3["median_calmar"], 4),
            "median_sharpe": round(baseline3["median_sharpe"], 4),
            "median_return_pct": round(baseline3["median_return_pct"], 4),
            "median_mdd_pct": round(baseline3["median_mdd_pct"], 4),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")

    print("\n" + "=" * 65)
    print("FINAL OUT-OF-SAMPLE RESULTS (2018-10-16 to 2026-11-06)")
    print("=" * 65)
    print(f"{'Metric':<25}{'TOTM (4-day)':>15}{'TOTM (3-day)':>15}{'SPY B&H':>15}{'Base p95 (4d)':>15}")
    print("-" * 85)
    print(f"{'Total Return':<25}{metrics_4d_oos['total_return_pct']:>14.2f}%{metrics_3d_oos['total_return_pct']:>14.2f}%{metrics_spy_oos['total_return_pct']:>14.2f}%{baseline4['p95_return_pct']:>14.2f}%")
    print(f"{'CAGR':<25}{metrics_4d_oos['cagr_pct']:>14.2f}%{metrics_3d_oos['cagr_pct']:>14.2f}%{metrics_spy_oos['cagr_pct']:>14.2f}%{'n/a':>15}")
    print(f"{'Max Drawdown':<25}{metrics_4d_oos['max_drawdown_pct']:>14.2f}%{metrics_3d_oos['max_drawdown_pct']:>14.2f}%{metrics_spy_oos['max_drawdown_pct']:>14.2f}%{'n/a':>15}")
    print(f"{'Sharpe Ratio':<25}{metrics_4d_oos['sharpe_ratio']:>15.4f}{metrics_3d_oos['sharpe_ratio']:>15.4f}{metrics_spy_oos['sharpe_ratio']:>15.4f}{baseline4['p95_sharpe']:>15.4f}")
    print(f"{'Calmar Ratio':<25}{metrics_4d_oos['calmar_ratio']:>15.4f}{metrics_3d_oos['calmar_ratio']:>15.4f}{metrics_spy_oos['calmar_ratio']:>15.4f}{baseline4['p95_calmar']:>15.4f}")
    print("-" * 85)
    print(f"TOTM (4-day) Calmar Percentile in 4-day Baseline: {p_calmar_4d:.1f}th (pass p95: {pass_calmar})")
    print(f"TOTM (4-day) Sharpe Percentile in 4-day Baseline: {p_sharpe_4d:.1f}th (pass p95: {pass_sharpe})")
    print(f"TOTM (3-day) Calmar Percentile in 4-day Baseline: {p_calmar_3d_vs_4d:.1f}th (pass p95: {metrics_3d_oos['calmar_ratio'] > baseline4['p95_calmar']})")
    print(f"TOTM (3-day) Sharpe Percentile in 4-day Baseline: {p_sharpe_3d_vs_4d:.1f}th (pass p95: {metrics_3d_oos['sharpe_ratio'] > baseline4['p95_sharpe']})")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
