"""Intraday meta-strategy backtest simulator (frozen rules).

Read-only research. Implements the frozen meta-strategy: for each rolling
window, test the 4 strategies x 5 assets over a 30-day lookback, pick the
single highest-net-profit combination, trade it for the next 7 days, roll
forward by 7 days, and compare the final expectancy to a random baseline.

No parameter is tuned; the lookback (30 days) and trade window (7 days) are
frozen by the rules file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ("SPY", "QQQ", "TSLA", "AAPL", "BTC-USD")
TIMEFRAME = "15m"
LOOKBACK_DAYS = 30
TRADE_WINDOW_DAYS = 7
BAR_MINUTES = 15
COST_BPS = 10.0
STOP_PCT = 0.01
TP_PCT = 0.02
BASELINE_RUNS = 1000
BASELINE_PERCENTILE = 95.0
STRATEGIES = ("ma_crossover", "rsi", "bollinger", "vwap")


@dataclass(frozen=True)
class MetaTrade:
    asset: str
    strategy: str
    window_start: str
    entry_ts: str
    exit_ts: str
    entry: float
    exit: float
    ret: float
    hold_minutes: float


def fetch_bars(symbol: str) -> pd.DataFrame:
    frame = yf.Ticker(symbol).history(period="60d", interval=TIMEFRAME, auto_adjust=True)
    if frame is None or frame.empty:
        raise ValueError(f"no {TIMEFRAME} bars for {symbol}")
    out = frame.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    missing = [column for column in ("open", "high", "low", "close") if column not in out.columns]
    if missing:
        raise ValueError(f"{symbol} bars need columns {missing}")
    out = out.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    index = pd.DatetimeIndex(out.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    out.index = index
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    return out


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).mask((loss == 0) & gain.notna(), 100.0)


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    return _sma(typical, n)


def _signals(bars: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series(np.full(len(bars), 1000.0), index=bars.index)

    if name == "ma_crossover":
        fast = _sma(close, 20)
        slow = _sma(close, 50)
        entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exit_ = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    elif name == "rsi":
        r = _rsi(close, 14)
        entry = (r > 30.0) & (r.shift(1) <= 30.0)
        exit_ = (r < 70.0) & (r.shift(1) >= 70.0)
    elif name == "bollinger":
        mid = _sma(close, 20)
        upper = mid + 2.0 * close.rolling(20).std()
        entry = (close > upper) & (close.shift(1) <= upper.shift(1))
        exit_ = (close < mid) & (close.shift(1) >= mid.shift(1))
    elif name == "vwap":
        v = _vwap(high, low, close, 20)
        fast = _ema(close, 9)
        liquid = volume > _sma(volume, 20)
        entry = (fast > v) & (fast.shift(1) <= v.shift(1)) & liquid
        exit_ = (fast < v) & (fast.shift(1) >= v.shift(1)) & liquid
    else:
        raise ValueError(f"unknown strategy: {name}")
    return np.asarray(entry.fillna(False).to_numpy(), dtype=bool), np.asarray(exit_.fillna(False).to_numpy(), dtype=bool)


def _cost_fraction(cost_bps: float) -> float:
    return (cost_bps / 10_000.0) / 2.0


def run_strategy(
    bars: pd.DataFrame,
    name: str,
    stop_pct: float = STOP_PCT,
    tp_pct: float = TP_PCT,
    cost_bps: float = COST_BPS,
) -> list[MetaTrade]:
    n = len(bars)
    entries, exits = _signals(bars, name)
    cost = _cost_fraction(cost_bps)
    trades: list[MetaTrade] = []
    pos: dict[str, Any] | None = None
    for i in range(n):
        if pos is None:
            if bool(entries[i]) and i + 1 < n:
                entry_fill = float(bars["open"].iloc[i + 1])
                pos = {
                    "entry_i": i,
                    "entry_fill": entry_fill,
                    "entry_ts": bars.index[i + 1],
                    "stop": entry_fill * (1.0 - stop_pct),
                    "tp": entry_fill * (1.0 + tp_pct),
                }
            continue
        price = float(bars["close"].iloc[i])
        if price <= pos["stop"] or price >= pos["tp"]:
            exit_fill = float(bars["open"].iloc[i + 1]) if i + 1 < n else price
            ret = exit_fill / pos["entry_fill"] - 1.0 - cost
            trades.append(
                MetaTrade(
                    asset=str(bars.attrs.get("asset", "")),
                    strategy=name,
                    window_start="",
                    entry_ts=str(pos["entry_ts"]),
                    exit_ts=str(bars.index[i + 1] if i + 1 < n else bars.index[i]),
                    entry=pos["entry_fill"],
                    exit=exit_fill,
                    ret=float(ret * 100.0),
                    hold_minutes=float(((bars.index[i + 1] if i + 1 < n else bars.index[i]) - pos["entry_ts"]).total_seconds() / 60.0),
                )
            )
            pos = None
    if pos is not None:
        last_ts = bars.index[-1]
        ret = float(bars["close"].iloc[-1]) / pos["entry_fill"] - 1.0 - cost
        trades.append(
            MetaTrade(
                asset=str(bars.attrs.get("asset", "")),
                strategy=name,
                window_start="",
                entry_ts=str(pos["entry_ts"]),
                exit_ts=str(last_ts),
                entry=pos["entry_fill"],
                exit=float(bars["close"].iloc[-1]),
                ret=float(ret * 100.0),
                hold_minutes=float((last_ts - pos["entry_ts"]).total_seconds() / 60.0),
            )
        )
    return trades


def summarize(trades: list[MetaTrade]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "expectancy_pct": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0}
    rets = np.array([trade.ret for trade in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    equity = np.concatenate(([1.0], np.cumprod(1.0 + rets / 100.0)))
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min() * 100.0)
    total_return = float((equity[-1] - 1.0) * 100.0)
    return {
        "trades": int(len(rets)),
        "win_rate": float((rets > 0).mean()),
        "expectancy_pct": float(rets.mean()),
        "max_drawdown_pct": max_dd,
        "total_return_pct": total_return,
    }


def random_baseline(bars: pd.DataFrame, holds: np.ndarray, cost_bps: float, asset: str, strategy: str, window_start: str) -> dict[str, Any]:
    n = len(bars)
    cost = _cost_fraction(cost_bps)
    rng = np.random.default_rng(20261105)
    values: list[float] = []
    for _ in range(BASELINE_RUNS):
        entry_i = int(rng.integers(0, max(n - int(holds.max()) - 2, 1), size=1)[0]) if n > 2 else 0
        hold = float(rng.choice(holds, size=1)[0])
        entry_fill = float(bars["open"].iloc[entry_i + 1]) if entry_i + 1 < n else float(bars["close"].iloc[-1])
        exit_fill = float(bars["open"].iloc[entry_i + 1 + int(hold)]) if entry_i + 1 + int(hold) < n else float(bars["close"].iloc[-1])
        values.append(float((exit_fill / entry_fill - 1.0 - cost) * 100.0))
    arr = np.array(values, dtype=float)
    return {
        "runs": BASELINE_RUNS,
        "percentile": float(np.percentile(arr, BASELINE_PERCENTILE)),
        "mean_pct": float(arr.mean()),
    }


def main() -> int:
    report: dict[str, Any] = {
        "strategy": "intraday_meta_strategy_backtest",
        "rules_file": "research/meta_strategy_rules.md",
        "assets": list(ASSETS),
        "timeframe": TIMEFRAME,
        "bar_minutes": BAR_MINUTES,
        "lookback_days": LOOKBACK_DAYS,
        "trade_window_days": TRADE_WINDOW_DAYS,
        "cost_bps": COST_BPS,
        "stop_pct": STOP_PCT,
        "take_profit_pct": TP_PCT,
        "baseline": {"runs": BASELINE_RUNS, "percentile": BASELINE_PERCENTILE},
    }
    windows: list[dict[str, Any]] = []
    switch_count = 0
    prev_key: tuple[str, str] | None = None
    for asset in ASSETS:
        bars = fetch_bars(asset)
        bars.attrs["asset"] = asset
        first = bars.index.min().date()
        last = bars.index.max().date()
        start = first
        while start + timedelta(days=TRADE_WINDOW_DAYS) <= last + timedelta(days=1):
            window_end = start + timedelta(days=TRADE_WINDOW_DAYS)
            lookback_end = start
            lookback_start = start - timedelta(days=LOOKBACK_DAYS)
            lb = bars.loc[(bars.index >= pd.Timestamp(lookback_start, tz="UTC")) & (bars.index < pd.Timestamp(lookback_end, tz="UTC"))]
            best_name = None
            best_net = -float("inf")
            for name in STRATEGIES:
                trades = run_strategy(lb, name, stop_pct=STOP_PCT, tp_pct=TP_PCT, cost_bps=COST_BPS)
                summ = summarize(trades)
                if float(summ["total_return_pct"]) > best_net:
                    best_net = float(summ["total_return_pct"])
                    best_name = name
            window_trades = run_strategy(bars.loc[(bars.index >= pd.Timestamp(start, tz="UTC")) & (bars.index < pd.Timestamp(window_end, tz="UTC"))], best_name, stop_pct=STOP_PCT, tp_pct=TP_PCT, cost_bps=COST_BPS)
            wm = summarize(window_trades)
            key = (asset, best_name)
            if prev_key is not None and key != prev_key:
                switch_count += 1
            prev_key = key
            windows.append(
                {
                    "asset": asset,
                    "strategy": best_name,
                    "window_start": str(start),
                    "window_end": str(window_end),
                    "net_profit_pct": best_net,
                    "metrics": wm,
                    "trades": [asdict(trade) for trade in window_trades],
                    "holds_bars": [int(round(getattr(trade, "hold_minutes") / 15.0)) for trade in window_trades],
                }
            )
            start = window_end
    report["windows"] = windows
    report["switch_count"] = switch_count
    report["final_window"] = windows[-1] if windows else None
    final = windows[-1] if windows else None
    final_baseline = None
    if final is not None:
        final_bars = fetch_bars(str(final["asset"]))
        final_baseline = random_baseline(
            final_bars,
            np.array([float(hold) for hold in final["holds_bars"]], dtype=float),
            COST_BPS,
            str(final["asset"]),
            str(final["strategy"]),
            str(final["window_start"]),
        )
        report["final_baseline"] = final_baseline
        report["final_expectancy_pct"] = final["metrics"]["expectancy_pct"]
    out = Path("research/meta_backtest_report.json")
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"report={out}")
    print(f"windows={len(windows)} switches={switch_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
