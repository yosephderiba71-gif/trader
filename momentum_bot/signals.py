"""Multi-asset time-series momentum signals.

The rules are frozen in ``research/momentum_rules.md``. This module only
computes data and target weights; it never talks to a broker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ASSETS = ("SPY", "TLT", "GLD")
SMA_WINDOW = 200
MOMENTUM_WINDOW = 60
VOL_WINDOW = 60
TOP_K = 2
MAX_WEIGHT = 0.50
TRADING_DAYS = 252
CASH = "CASH"


@dataclass(frozen=True)
class AssetMetrics:
    symbol: str
    close: float
    sma200: float
    eligible: bool
    return_60d: float
    vol_60d: float


def fetch_prices(period: str = "2y") -> pd.DataFrame:
    """Download adjusted daily closes for the frozen universe."""
    import yfinance as yf

    raw = yf.download(
        list(ASSETS),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no price data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        if len(ASSETS) == 1:
            close.columns = [ASSETS[0]]

    missing = [symbol for symbol in ASSETS if symbol not in close.columns]
    if missing:
        raise RuntimeError(f"Missing price data for: {', '.join(missing)}")

    return close.loc[:, list(ASSETS)].sort_index()


def compute_metrics(prices: pd.DataFrame) -> list[AssetMetrics]:
    metrics: list[AssetMetrics] = []
    for symbol in ASSETS:
        if symbol not in prices.columns:
            continue
        series = pd.to_numeric(prices[symbol], errors="coerce").dropna()
        if len(series) < SMA_WINDOW:
            metrics.append(
                AssetMetrics(symbol, float("nan"), float("nan"), False, float("nan"), float("nan"))
            )
            continue

        close = float(series.iloc[-1])
        sma200 = float(series.iloc[-SMA_WINDOW:].mean())
        return_60d = float(close / float(series.iloc[-1 - MOMENTUM_WINDOW]) - 1.0)
        log_returns = np.log(series).diff().iloc[-VOL_WINDOW:]
        vol_60d = float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        eligible = bool(
            close > sma200 and np.isfinite(return_60d) and np.isfinite(vol_60d)
        )
        metrics.append(AssetMetrics(symbol, close, sma200, eligible, return_60d, vol_60d))
    return metrics


def rank_assets(metrics: list[AssetMetrics], top_k: int = TOP_K) -> list[str]:
    eligible = [metric for metric in metrics if metric.eligible]
    eligible.sort(
        key=lambda metric: (
            -round(metric.return_60d, 12),
            round(metric.vol_60d, 12),
            metric.symbol,
        )
    )
    return [metric.symbol for metric in eligible[:top_k]]


def _apply_weight_cap(raw: dict[str, float], max_weight: float) -> dict[str, float]:
    weights: dict[str, float] = {}
    active = dict(raw)
    remaining = 1.0
    while active:
        total = sum(active.values())
        scaled = {symbol: remaining * weight / total for symbol, weight in active.items()}
        over = [symbol for symbol, weight in scaled.items() if weight > max_weight + 1e-12]
        if not over:
            weights.update(scaled)
            break
        for symbol in over:
            weights[symbol] = max_weight
            remaining -= max_weight
            del active[symbol]
        if remaining < -1e-12:
            raise ValueError("max_weight leaves no room for the remaining assets")
    return weights


def volatility_weights(
    vols: dict[str, float], max_weight: float = MAX_WEIGHT
) -> dict[str, float]:
    symbols = list(vols)
    if not symbols:
        return {CASH: 1.0}

    valid = all(vol is not None and np.isfinite(vol) and vol > 0 for vol in vols.values())
    if valid:
        inverse = {symbol: 1.0 / float(vol) for symbol, vol in vols.items()}
        total = sum(inverse.values())
        raw = {symbol: inverse[symbol] / total for symbol in symbols}
    else:
        raw = {symbol: 1.0 / len(symbols) for symbol in symbols}

    weights = _apply_weight_cap(raw, max_weight)
    invested = sum(weights.values())
    if invested < 1.0 - 1e-12:
        weights[CASH] = 1.0 - invested
    return weights


def compute_target_weights(prices: pd.DataFrame) -> dict[str, float]:
    metrics = compute_metrics(prices)
    selected = rank_assets(metrics)
    if not selected:
        return {CASH: 1.0}

    by_symbol = {metric.symbol: metric for metric in metrics}
    vols = {symbol: by_symbol[symbol].vol_60d for symbol in selected}
    weights = volatility_weights(vols)
    weights.setdefault(CASH, 0.0)
    return weights


def latest_prices(prices: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for symbol in ASSETS:
        if symbol not in prices.columns:
            continue
        series = pd.to_numeric(prices[symbol], errors="coerce").dropna()
        if not series.empty:
            result[symbol] = float(series.iloc[-1])
    return result


def format_report(metrics: list[AssetMetrics], targets: dict[str, float]) -> str:
    lines = ["Momentum signal snapshot:"]
    lines.append(
        f"{'SYMBOL':<8}{'CLOSE':>10}{'SMA200':>10}{'ELIGIBLE':>10}{'60D RET':>10}{'60D VOL':>10}"
    )
    for metric in metrics:
        lines.append(
            f"{metric.symbol:<8}{metric.close:>10.2f}{metric.sma200:>10.2f}"
            f"{str(metric.eligible):>10}{metric.return_60d:>10.2%}{metric.vol_60d:>10.2%}"
        )
    lines.append(
        "Target weights: "
        + ", ".join(f"{symbol} {weight:.2%}" for symbol, weight in targets.items())
    )
    return "\n".join(lines)
