"""Cross-sectional sector momentum signals.

``compute_target_weights`` implements the Trial #24 unlevered rules, which are
pending final review. The volatility-targeting helpers are retained only for
the rejected Trial #25 research record and must not be used for live execution.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

SECTOR_ASSETS = (
    "XLK",
    "XLF",
    "XLV",
    "XLE",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
)
SMA_WINDOW = 200
SKIP_DAYS = 20
MOMENTUM_WINDOW = 120
TOP_K = 3
CASH = "CASH"
VOL_WINDOW = 60
TARGET_VOL = 0.15
MIN_SCALAR = 0.50
MAX_SCALAR = 1.35
TRADING_DAYS = 252
MARGIN_RATE = 0.05


@dataclass(frozen=True)
class SectorMetrics:
    symbol: str
    close: float
    sma200: float
    eligible: bool
    skip_return_120d: float


@dataclass(frozen=True)
class VolTargetDecision:
    selected: tuple[str, ...]
    basket_volatility: float
    scalar: float
    weights: dict[str, float]
    gross_exposure: float
    margin_debit: float


def _empty_metrics(symbol: str) -> SectorMetrics:
    return SectorMetrics(
        symbol=symbol,
        close=float("nan"),
        sma200=float("nan"),
        eligible=False,
        skip_return_120d=float("nan"),
    )


def compute_metrics(prices: pd.DataFrame) -> list[SectorMetrics]:
    metrics: list[SectorMetrics] = []
    minimum_observations = max(SMA_WINDOW, MOMENTUM_WINDOW + 1)

    for symbol in SECTOR_ASSETS:
        if symbol not in prices.columns:
            metrics.append(_empty_metrics(symbol))
            continue

        series = pd.to_numeric(prices[symbol], errors="coerce").dropna()
        if len(series) < minimum_observations:
            metrics.append(_empty_metrics(symbol))
            continue

        close = float(series.iloc[-1])
        sma200 = float(series.iloc[-SMA_WINDOW:].mean())
        momentum_base = float(series.iloc[-1 - MOMENTUM_WINDOW])
        skip_price = float(series.iloc[-1 - SKIP_DAYS])
        skip_return = (
            float(skip_price / momentum_base - 1.0)
            if momentum_base > 0.0
            else float("nan")
        )
        eligible = bool(close > sma200 and np.isfinite(skip_return))
        metrics.append(
            SectorMetrics(
                symbol=symbol,
                close=close,
                sma200=sma200,
                eligible=eligible,
                skip_return_120d=skip_return,
            )
        )

    return metrics


def rank_sectors(
    metrics: list[SectorMetrics],
    top_k: int = TOP_K,
) -> list[str]:
    eligible = [
        metric
        for metric in metrics
        if metric.eligible and np.isfinite(metric.skip_return_120d)
    ]
    eligible.sort(key=lambda metric: (-metric.skip_return_120d, metric.symbol))
    return [metric.symbol for metric in eligible[:top_k]]


def compute_target_weights(prices: pd.DataFrame) -> dict[str, float]:
    selected = rank_sectors(compute_metrics(prices))
    weights = {symbol: 1.0 / TOP_K for symbol in selected}
    weights[CASH] = max(0.0, 1.0 - len(selected) / TOP_K)
    return weights


def compute_basket_volatility(
    prices: pd.DataFrame,
    selected: Sequence[str],
    window: int = VOL_WINDOW,
) -> float:
    """Annualized volatility of the equal-weight selected-sector basket.

    Each selected sector receives 1/TOP_K. Any unallocated remainder is cash
    with a zero return, so a two-sector basket earns two thirds of the average
    selected-sector return and a one-sector basket earns one third.
    """
    symbols = list(selected)
    if not symbols:
        return 0.0
    if any(symbol not in prices.columns for symbol in symbols):
        return float("nan")

    frame = prices.loc[:, symbols].apply(pd.to_numeric, errors="coerce")
    returns = frame.pct_change(fill_method=None).iloc[-window:].dropna(how="any")
    if len(returns) < 2:
        return float("nan")

    basket_returns = returns.sum(axis=1) / TOP_K
    volatility = float(basket_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return volatility if np.isfinite(volatility) else float("nan")


def volatility_scalar(
    basket_volatility: float,
    has_positions: bool = True,
) -> float:
    """Map basket volatility to the bounded Trial #25 scalar."""
    if not has_positions:
        return 1.0
    if not np.isfinite(basket_volatility):
        return 1.0
    if basket_volatility <= 0.0:
        return MAX_SCALAR
    return float(np.clip(TARGET_VOL / basket_volatility, MIN_SCALAR, MAX_SCALAR))


def compute_vol_target_decision(prices: pd.DataFrame) -> VolTargetDecision:
    selected = tuple(rank_sectors(compute_metrics(prices)))
    basket_volatility = compute_basket_volatility(prices, selected)
    scalar = volatility_scalar(basket_volatility, has_positions=bool(selected))

    if selected:
        weights = {symbol: scalar / TOP_K for symbol in selected}
        gross_exposure = scalar * len(selected) / TOP_K
        weights[CASH] = 1.0 - gross_exposure
    else:
        weights = {CASH: 1.0}
        gross_exposure = 0.0

    return VolTargetDecision(
        selected=selected,
        basket_volatility=basket_volatility,
        scalar=scalar,
        weights=weights,
        gross_exposure=gross_exposure,
        margin_debit=max(0.0, gross_exposure - 1.0),
    )


def compute_vol_target_weights(prices: pd.DataFrame) -> dict[str, float]:
    return compute_vol_target_decision(prices).weights
