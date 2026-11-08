"""Unit tests for the frozen Trial #24 core and Trial #25 vol-target rules.

Expected implementation module: ``momentum_bot.sector_momentum``.

Deterministic math only: no network, no Alpaca calls, no flat files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momentum_bot.run_live import format_report
from momentum_bot.sector_momentum import (
    CASH,
    MAX_SCALAR,
    MIN_SCALAR,
    SECTOR_ASSETS,
    TOP_K,
    VOL_WINDOW,
    SectorMetrics,
    compute_basket_volatility,
    compute_metrics,
    compute_target_weights,
    compute_vol_target_decision,
    volatility_scalar,
)
from research.run_sector_momentum import (
    calmar_ratio,
    cagr,
    max_drawdown,
    sharpe_ratio,
)

BASE = 100.0
SMA_WINDOW = 200
MOMENTUM_WINDOW = 120
SKIP_DAYS = 20


def _skip_month_path(skip_return: float, last_20_return: float = 0.0) -> np.ndarray:
    """320 closes with a known skip-month return and a separate final 20-day move."""
    pre = np.full(SMA_WINDOW, BASE)
    first_leg = np.linspace(
        BASE,
        BASE * (1.0 + skip_return),
        MOMENTUM_WINDOW - SKIP_DAYS,
    )
    second_leg = np.linspace(
        first_leg[-1],
        first_leg[-1] * (1.0 + last_20_return),
        SKIP_DAYS,
    )
    return np.concatenate([pre, first_leg, second_leg])


def _below_sma_path(final_close: float, peak: float = 300.0) -> np.ndarray:
    """320 closes: 200 flat days, a 100-day rise, then a 20-day fall below SMA200."""
    pre = np.full(SMA_WINDOW, BASE)
    rise = np.linspace(BASE, peak, 100)
    fall = np.linspace(peak, final_close, SKIP_DAYS)
    return np.concatenate([pre, rise, fall])


def _frame(
    paths: dict[str, np.ndarray],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    if require_all:
        assert set(paths) == set(SECTOR_ASSETS)
    else:
        assert set(paths) <= set(SECTOR_ASSETS)
    length = len(next(iter(paths.values())))
    assert all(len(path) == length for path in paths.values())
    index = pd.bdate_range("2024-01-01", periods=length)
    return pd.DataFrame(paths, index=index)


def _prices_from_returns(returns_by_symbol: dict[str, np.ndarray]) -> pd.DataFrame:
    length = len(next(iter(returns_by_symbol.values())))
    assert all(len(returns) == length for returns in returns_by_symbol.values())
    index = pd.bdate_range("2024-01-01", periods=length + 1)
    paths = {
        symbol: BASE * np.concatenate([[1.0], np.cumprod(1.0 + returns)])
        for symbol, returns in returns_by_symbol.items()
    }
    return pd.DataFrame(paths, index=index)


def _vol_path(drift: float, daily_vol: float) -> np.ndarray:
    pre = np.full(SMA_WINDOW, BASE)
    returns = np.empty(MOMENTUM_WINDOW)
    returns[0::2] = drift + daily_vol
    returns[1::2] = drift - daily_vol
    return np.concatenate([pre, BASE * np.cumprod(1.0 + returns)])


def _vol_frame(drift: float, daily_vol: float) -> pd.DataFrame:
    drifts = [drift - 0.0002 * offset for offset in range(len(SECTOR_ASSETS))]
    return _frame(
        {
            symbol: _vol_path(symbol_drift, daily_vol)
            for symbol, symbol_drift in zip(SECTOR_ASSETS, drifts)
        }
    )


def _assert_above_sma200(path: np.ndarray) -> None:
    sma200 = float(pd.Series(path).iloc[-SMA_WINDOW:].mean())
    assert path[-1] > sma200


def _assert_below_sma200(path: np.ndarray) -> None:
    sma200 = float(pd.Series(path).iloc[-SMA_WINDOW:].mean())
    assert path[-1] < sma200


def _selected(targets: dict[str, float]) -> set[str]:
    return {
        symbol
        for symbol, weight in targets.items()
        if symbol != CASH and weight > 1e-12
    }


def test_universe_is_all_eleven_spdr_sectors():
    assert SECTOR_ASSETS == (
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
    assert len(SECTOR_ASSETS) == 11
    assert "SMH" not in SECTOR_ASSETS


def test_crash_gate_drops_below_sma200_sector_and_pulls_in_next_ranked():
    below = _below_sma_path(150.0)
    paths = {"XLK": below}
    for offset, symbol in enumerate(SECTOR_ASSETS[1:]):
        paths[symbol] = _skip_month_path(0.40 - 0.03 * offset)

    _assert_below_sma200(below)
    for symbol in SECTOR_ASSETS[1:]:
        _assert_above_sma200(paths[symbol])

    frame = _frame(paths)
    metrics = {metric.symbol: metric for metric in compute_metrics(frame)}
    targets = compute_target_weights(frame)

    assert metrics["XLK"].eligible is False
    assert "XLK" not in targets
    assert _selected(targets) == {"XLF", "XLV", "XLE"}
    for symbol in ("XLF", "XLV", "XLE"):
        assert targets[symbol] == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_all_sectors_below_sma200_returns_100_percent_cash():
    finals = [130.0, 125.0, 120.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0, 85.0, 80.0]
    paths = {
        symbol: _below_sma_path(final)
        for symbol, final in zip(SECTOR_ASSETS, finals)
    }
    for path in paths.values():
        _assert_below_sma200(path)

    targets = compute_target_weights(_frame(paths))

    assert targets.get(CASH, 0.0) == pytest.approx(1.0)
    assert sum(
        weight for symbol, weight in targets.items() if symbol != CASH
    ) == pytest.approx(0.0, abs=1e-12)


def test_skip_month_return_ignores_last_20_days():
    returns = {
        "XLK": (0.00, 1.00),
        "XLF": (0.20, 0.00),
        "XLV": (0.30, 0.00),
        "XLE": (0.40, 0.00),
        "XLI": (0.10, 0.00),
        "XLY": (0.09, 0.00),
        "XLP": (0.08, 0.00),
        "XLU": (0.07, 0.00),
        "XLB": (0.06, 0.00),
        "XLRE": (0.05, 0.00),
        "XLC": (0.04, 0.00),
    }
    paths = {
        symbol: _skip_month_path(skip_return, last_20_return)
        for symbol, (skip_return, last_20_return) in returns.items()
    }
    frame = _frame(paths)
    metrics = {metric.symbol: metric for metric in compute_metrics(frame)}

    assert metrics["XLK"].skip_return_120d == pytest.approx(0.0, abs=1e-12)
    assert metrics["XLK"].close == pytest.approx(200.0)

    targets = compute_target_weights(frame)

    assert _selected(targets) == {"XLE", "XLV", "XLF"}
    assert "XLK" not in targets


def test_missing_sector_column_is_ineligible_and_does_not_break():
    paths = {
        symbol: _skip_month_path(0.10 + 0.01 * offset)
        for offset, symbol in enumerate(SECTOR_ASSETS)
        if symbol != "XLC"
    }
    frame = _frame(paths, require_all=False)
    metrics = {metric.symbol: metric for metric in compute_metrics(frame)}
    targets = compute_target_weights(frame)

    assert metrics["XLC"].eligible is False
    assert "XLC" not in targets
    assert len(_selected(targets)) == 3


def test_sector_with_insufficient_history_is_ineligible():
    paths = {
        symbol: _skip_month_path(0.10 + 0.01 * offset)
        for offset, symbol in enumerate(SECTOR_ASSETS)
    }
    paths["XLC"] = _skip_month_path(1.00)
    frame = _frame(paths)
    frame.loc[frame.index[:180], "XLC"] = np.nan

    metrics = {metric.symbol: metric for metric in compute_metrics(frame)}
    targets = compute_target_weights(frame)

    assert metrics["XLC"].eligible is False
    assert "XLC" not in targets
    assert len(_selected(targets)) == 3


def test_fewer_than_three_eligible_holds_cash_remainder():
    paths = {
        "XLK": _skip_month_path(0.30),
        "XLF": _skip_month_path(0.20),
    }
    finals = [130.0, 125.0, 120.0, 115.0, 110.0, 105.0, 100.0, 95.0, 90.0]
    below_symbols = [symbol for symbol in SECTOR_ASSETS if symbol not in paths]
    for symbol, final in zip(below_symbols, finals):
        paths[symbol] = _below_sma_path(final)

    targets = compute_target_weights(_frame(paths))

    assert _selected(targets) == {"XLK", "XLF"}
    for symbol in ("XLK", "XLF"):
        assert targets[symbol] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert targets.get(CASH, 0.0) == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_top_three_get_equal_thirds():
    returns = [0.50, 0.40, 0.30, 0.20, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
    paths = {
        symbol: _skip_month_path(value)
        for symbol, value in zip(SECTOR_ASSETS, returns)
    }

    targets = compute_target_weights(_frame(paths))
    selected = _selected(targets)

    assert selected == {"XLK", "XLF", "XLV"}
    for symbol in selected:
        assert targets[symbol] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert sum(targets[symbol] for symbol in selected) == pytest.approx(1.0, abs=1e-4)
    assert targets.get(CASH, 0.0) == pytest.approx(0.0, abs=1e-4)


def test_basket_volatility_uses_equal_weight_portfolio_returns():
    alternating = np.tile([0.02, -0.02], VOL_WINDOW // 2)
    frame = _prices_from_returns(
        {
            "XLK": alternating,
            "XLF": -alternating,
            "XLV": np.zeros(VOL_WINDOW),
        }
    )

    basket_vol = compute_basket_volatility(frame, ("XLK", "XLF", "XLV"))
    individual_vols = [
        frame[symbol].pct_change(fill_method=None).dropna().std(ddof=1)
        * np.sqrt(252.0)
        for symbol in ("XLK", "XLF", "XLV")
    ]

    assert float(np.mean(individual_vols)) > 0.10
    assert basket_vol == pytest.approx(0.0, abs=1e-12)


def test_volatility_scalar_formula_and_bounds():
    assert volatility_scalar(0.30) == pytest.approx(0.50)
    assert volatility_scalar(0.20) == pytest.approx(0.75)
    assert volatility_scalar(0.15) == pytest.approx(1.00)
    assert volatility_scalar(0.10) == pytest.approx(MAX_SCALAR)
    assert volatility_scalar(0.0) == pytest.approx(MAX_SCALAR)
    assert volatility_scalar(float("nan")) == pytest.approx(1.0)
    assert volatility_scalar(0.0, has_positions=False) == pytest.approx(1.0)


def test_vol_target_weights_apply_scalar_and_allow_negative_cash():
    frame = _vol_frame(drift=0.0015, daily_vol=0.003)
    for symbol in ("XLK", "XLF", "XLV"):
        _assert_above_sma200(frame[symbol].to_numpy())

    decision = compute_vol_target_decision(frame)

    assert decision.selected == ("XLK", "XLF", "XLV")
    assert decision.scalar == pytest.approx(MAX_SCALAR)
    for symbol in decision.selected:
        assert decision.weights[symbol] == pytest.approx(MAX_SCALAR / TOP_K)
    assert decision.weights[CASH] == pytest.approx(1.0 - MAX_SCALAR)
    assert decision.gross_exposure == pytest.approx(MAX_SCALAR)
    assert decision.margin_debit == pytest.approx(MAX_SCALAR - 1.0)


def test_vol_target_weights_de_lever_when_basket_vol_is_high():
    frame = _vol_frame(drift=0.0010, daily_vol=0.020)
    for symbol in ("XLK", "XLF", "XLV"):
        _assert_above_sma200(frame[symbol].to_numpy())

    decision = compute_vol_target_decision(frame)

    assert decision.selected == ("XLK", "XLF", "XLV")
    assert decision.scalar == pytest.approx(MIN_SCALAR)
    for symbol in decision.selected:
        assert decision.weights[symbol] == pytest.approx(MIN_SCALAR / TOP_K)
    assert decision.weights[CASH] == pytest.approx(1.0 - MIN_SCALAR)
    assert decision.gross_exposure == pytest.approx(MIN_SCALAR)
    assert decision.margin_debit == pytest.approx(0.0)


def test_format_report_marks_missing_data_as_na():
    metrics = [
        SectorMetrics("XLK", float("nan"), float("nan"), False, float("nan")),
        SectorMetrics("XLF", 100.0, 90.0, True, 0.05),
    ]

    report = format_report(metrics, {"XLF": 1.0, CASH: 0.0}, benchmark_close=100.0)

    assert "nan" not in report.lower()
    assert "n/a" in report
    assert "WARNING: missing price data for XLK" in report


def test_oos_metrics_definitions():
    equity = pd.Series(
        [100.0, 120.0, 90.0, 120.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )

    assert max_drawdown(equity) == pytest.approx(-25.0)
    expected_calmar = (
        cagr(100.0, 120.0, equity.index[0], equity.index[-1]) / 100.0
    ) / 0.25
    assert calmar_ratio(equity) == pytest.approx(expected_calmar)
    assert sharpe_ratio(
        pd.Series(
            [100.0, 100.0, 100.0, 100.0],
            index=pd.bdate_range("2024-01-01", periods=4),
        )
    ) == pytest.approx(0.0)
