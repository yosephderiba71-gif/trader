"""Unit tests for the frozen multi-asset time-series momentum rules.

Deterministic math only: no network, no Alpaca calls, no flat files.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from momentum_bot.rebalance import RebalanceOrder, compute_rebalance_orders, rebalance
from momentum_bot.signals import (
    MOMENTUM_WINDOW,
    AssetMetrics,
    compute_target_weights,
    rank_assets,
    volatility_weights,
)


def _path_with_60d_return(seed: int, target_return: float, vol: float, end_base: float = 130.0) -> np.ndarray:
    """260 closes whose last 60 log returns have ~vol and sum to target_return."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, vol, MOMENTUM_WINDOW)
    returns += (math.log1p(target_return) - returns.sum()) / MOMENTUM_WINDOW
    pre = np.linspace(end_base * 0.6, end_base, 200)
    tail = end_base * np.exp(np.cumsum(returns))
    return np.concatenate([pre, tail])


def _frame(series_by_symbol: dict[str, np.ndarray]) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(next(iter(series_by_symbol.values()))))
    return pd.DataFrame(series_by_symbol, index=index)


def _metric(symbol: str, ret: float, vol: float, eligible: bool = True) -> AssetMetrics:
    return AssetMetrics(
        symbol=symbol,
        close=100.0,
        sma200=90.0 if eligible else 110.0,
        eligible=eligible,
        return_60d=ret,
        vol_60d=vol,
    )


def _below_sma(path: np.ndarray, factor: float = 0.80) -> np.ndarray:
    path = path.copy()
    sma200 = pd.Series(path).rolling(200).mean().iloc[-1]
    path[-5:] = sma200 * factor
    return path


def test_sma200_filter_excludes_asset_below_sma():
    frame = _frame(
        {
            "SPY": _path_with_60d_return(1, 0.12, 0.02),
            "TLT": _path_with_60d_return(2, 0.08, 0.02),
            "GLD": _below_sma(_path_with_60d_return(3, 0.10, 0.02)),
        }
    )

    targets = compute_target_weights(frame)

    assert "GLD" not in targets
    assert set(targets) == {"SPY", "TLT", "CASH"}


def test_ranking_sorts_by_60d_return_and_takes_top_two():
    frame = _frame(
        {
            "SPY": _path_with_60d_return(11, 0.15, 0.02),
            "TLT": _path_with_60d_return(12, 0.08, 0.02),
            "GLD": _path_with_60d_return(13, 0.04, 0.02),
        }
    )

    targets = compute_target_weights(frame)

    assert set(targets) == {"SPY", "TLT", "CASH"}
    assert targets["CASH"] == pytest.approx(0.0)


def test_ranking_tie_breaker_uses_lowest_vol():
    metrics = [
        _metric("SPY", 0.10, 0.03),
        _metric("TLT", 0.10, 0.01),
        _metric("GLD", 0.10, 0.02),
    ]

    assert rank_assets(metrics) == ["TLT", "GLD"]


def test_ranking_final_tie_breaker_is_alphabetical():
    metrics = [
        _metric("SPY", 0.10, 0.02),
        _metric("TLT", 0.05, 0.02),
        _metric("GLD", 0.10, 0.02),
    ]

    assert rank_assets(metrics) == ["GLD", "SPY"]


def test_zero_eligible_assets_returns_100_percent_cash():
    frame = _frame(
        {
            "SPY": _below_sma(_path_with_60d_return(21, 0.05, 0.02), 0.70),
            "TLT": _below_sma(_path_with_60d_return(22, 0.05, 0.02), 0.70),
            "GLD": _below_sma(_path_with_60d_return(23, 0.05, 0.02), 0.70),
        }
    )

    assert compute_target_weights(frame) == {"CASH": 1.0}


def test_insufficient_history_is_treated_as_ineligible():
    index = pd.bdate_range("2024-01-01", periods=100)
    frame = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 120.0, 100),
            "TLT": np.linspace(100.0, 110.0, 100),
            "GLD": np.linspace(100.0, 105.0, 100),
        },
        index=index,
    )

    assert compute_target_weights(frame) == {"CASH": 1.0}


def test_inverse_vol_weights_and_50pct_cap():
    weights = volatility_weights({"SPY": 0.10, "GLD": 0.20})

    assert weights["SPY"] == pytest.approx(0.50)
    assert weights["GLD"] == pytest.approx(0.50)
    assert weights.get("CASH", 0.0) == pytest.approx(0.0)


def test_inverse_vol_weights_without_cap():
    weights = volatility_weights({"SPY": 0.10, "TLT": 0.20, "GLD": 0.20})

    assert weights["SPY"] == pytest.approx(0.50)
    assert weights["TLT"] == pytest.approx(0.25)
    assert weights["GLD"] == pytest.approx(0.25)


def test_single_asset_is_half_invested_half_cash():
    weights = volatility_weights({"SPY": 0.15})

    assert weights["SPY"] == pytest.approx(0.50)
    assert weights["CASH"] == pytest.approx(0.50)


def test_target_weights_cap_and_redistribute():
    frame = _frame(
        {
            "SPY": _path_with_60d_return(31, 0.15, 0.02),
            "TLT": _below_sma(_path_with_60d_return(32, 0.05, 0.02), 0.75),
            "GLD": _path_with_60d_return(33, 0.10, 0.04),
        }
    )

    targets = compute_target_weights(frame)

    assert "TLT" not in targets
    assert targets["SPY"] == pytest.approx(0.50)
    assert targets["GLD"] == pytest.approx(0.50)
    assert targets["CASH"] == pytest.approx(0.0)


def test_rebalance_delta_sells_exits_first_then_buys():
    targets = {"SPY": 0.50, "GLD": 0.50, "CASH": 0.0}
    positions = {"TLT": 100.0, "SPY": 100.0}
    prices = {"SPY": 100.0, "GLD": 100.0, "TLT": 100.0}

    orders = compute_rebalance_orders(
        equity=100_000.0, targets=targets, positions=positions, prices=prices
    )

    assert [o.side for o in orders] == ["sell", "buy", "buy"]
    assert orders[0] == RebalanceOrder("TLT", "sell", 100.0, 100.0, 0.0)
    buys = {o.symbol: o.qty for o in orders if o.side == "buy"}
    assert buys == {"SPY": 400.0, "GLD": 500.0}


def test_rebalance_delta_trims_overweight_and_skips_noops():
    targets = {"SPY": 0.50, "CASH": 0.50}
    prices = {"SPY": 100.0}

    trim = compute_rebalance_orders(100_000.0, targets, {"SPY": 800.0}, prices)
    noop = compute_rebalance_orders(100_000.0, targets, {"SPY": 500.0}, prices)

    assert trim == [RebalanceOrder("SPY", "sell", 300.0, 800.0, 500.0)]
    assert noop == []


def test_rebalance_floors_target_shares():
    targets = {"SPY": 0.50, "CASH": 0.50}

    orders = compute_rebalance_orders(100_000.0, targets, {}, {"SPY": 333.0})

    assert orders == [RebalanceOrder("SPY", "buy", 150.0, 0.0, 150.0)]


def test_rebalance_requires_price_for_target():
    with pytest.raises(ValueError):
        compute_rebalance_orders(100_000.0, {"SPY": 0.50}, {}, {})


def test_rebalance_rejects_gross_exposure_above_100_percent():
    targets = {"SPY": 0.675, "GLD": 0.675, "CASH": -0.35}

    with pytest.raises(ValueError, match="target weights sum to more than 100%"):
        compute_rebalance_orders(
            equity=100_000.0,
            targets=targets,
            positions={},
            prices={"SPY": 100.0, "GLD": 100.0},
        )


def test_rebalance_sizes_from_equity_not_cash():
    class FakeAccount:
        equity = "100000"
        cash = "0"

    class FakeTradingClient:
        def get_account(self):
            return FakeAccount()

        def get_all_positions(self):
            return []

    orders = rebalance(
        {"SPY": 0.50, "GLD": 0.50, "CASH": 0.0},
        latest_prices={"SPY": 100.0, "GLD": 100.0},
        dry_run=True,
        trading_client=FakeTradingClient(),
        data_client=object(),
    )

    assert [order.symbol for order in orders] == ["GLD", "SPY"]
    assert all(order.qty == 500.0 for order in orders)


def test_rebalance_aborts_on_sell_timeout_and_skips_buys():
    from types import SimpleNamespace

    class FakeTradingClient:
        def __init__(self):
            self.submitted = []
            self.canceled = []

        def get_account(self):
            return SimpleNamespace(equity="100000")

        def get_all_positions(self):
            return [SimpleNamespace(symbol="TLT", qty="100", current_price="100")]

        def submit_order(self, order_data):
            self.submitted.append(order_data)
            return SimpleNamespace(
                id=f"order-{len(self.submitted)}",
                status="new",
                symbol=order_data.symbol,
            )

        def get_order_by_id(self, order_id):
            return SimpleNamespace(id=order_id, status="new")

        def cancel_order_by_id(self, order_id):
            self.canceled.append(order_id)

    client = FakeTradingClient()
    with pytest.raises(RuntimeError, match="sell order timeout"):
        rebalance(
            {"SPY": 0.50, "GLD": 0.50, "CASH": 0.0},
            latest_prices={"SPY": 100.0, "GLD": 100.0, "TLT": 100.0},
            dry_run=False,
            trading_client=client,
            data_client=object(),
            sell_fill_timeout=0.01,
            sell_fill_poll_interval=0.001,
        )

    assert [order.symbol for order in client.submitted] == ["TLT"]
    assert client.canceled == ["order-1"]


def test_rebalance_submits_buys_only_after_sells_fill():
    from types import SimpleNamespace

    class FakeTradingClient:
        def __init__(self):
            self.submitted = []

        def get_account(self):
            return SimpleNamespace(equity="100000")

        def get_all_positions(self):
            return [SimpleNamespace(symbol="TLT", qty="100", current_price="100")]

        def submit_order(self, order_data):
            self.submitted.append(order_data)
            return SimpleNamespace(
                id=f"order-{len(self.submitted)}",
                status="new",
                symbol=order_data.symbol,
            )

        def get_order_by_id(self, order_id):
            return SimpleNamespace(id=order_id, status="filled")

        def cancel_order_by_id(self, order_id):
            raise AssertionError("cancel should not be called")

    client = FakeTradingClient()
    orders = rebalance(
        {"SPY": 0.50, "GLD": 0.50, "CASH": 0.0},
        latest_prices={"SPY": 100.0, "GLD": 100.0, "TLT": 100.0},
        dry_run=False,
        trading_client=client,
        data_client=object(),
        sell_fill_timeout=0.01,
        sell_fill_poll_interval=0.001,
    )

    assert [order.side for order in orders] == ["sell", "buy", "buy"]
    assert [order.symbol for order in client.submitted] == ["TLT", "GLD", "SPY"]


def test_first_trading_day_guard():
    from datetime import date
    from types import SimpleNamespace

    from momentum_bot.rebalance import is_first_trading_day

    class FakeTradingClient:
        def __init__(self, days):
            self.days = days

        def get_calendar(self, request):
            return [SimpleNamespace(date=day) for day in self.days]

    assert is_first_trading_day(
        FakeTradingClient([date(2026, 11, 2)]),
        today=date(2026, 11, 2),
    )
    assert not is_first_trading_day(
        FakeTradingClient([date(2026, 11, 2), date(2026, 11, 3)]),
        today=date(2026, 11, 3),
    )
