from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.rsi2_spyqqq import (
    BASE_COST_BPS,
    STRESS_COST_BPS,
    Rsi2Trade,
    _net_return_pct,
    random_baseline,
    simulate,
    summarize,
)


def _daily_frame(closes: np.ndarray) -> pd.DataFrame:
    opens = np.r_[float(closes[0]), closes[:-1]]
    return pd.DataFrame(
        {
            "open": opens,
            "high": opens * 1.01,
            "low": opens * 0.99,
            "close": closes,
        },
        index=pd.date_range("2025-01-02", periods=len(closes), freq="D", tz="UTC"),
    )


def _declining_then_recovering_frame() -> pd.DataFrame:
    closes = np.empty(300)
    closes[:200] = np.linspace(50.0, 100.0, 200)
    closes[200:210] = np.linspace(100.0, 95.0, 10)
    closes[210:] = np.linspace(95.0, 130.0, len(closes) - 210)
    return _daily_frame(closes)


def test_net_return_pct_applies_buy_and_sell_costs() -> None:
    expected_base = (100.0 * (1.0 - 0.0005) / (100.0 * (1.0 + 0.0005)) - 1.0) * 100.0
    assert _net_return_pct(100.0, 100.0, BASE_COST_BPS) == pytest.approx(expected_base)
    expected_stress = (100.0 * (1.0 - 0.001) / (100.0 * (1.0 + 0.001)) - 1.0) * 100.0
    assert _net_return_pct(100.0, 100.0, STRESS_COST_BPS) == pytest.approx(expected_stress)


def test_simulate_finds_no_trade_before_sma200_warmup() -> None:
    frame = _daily_frame(np.full(50, 100.0))
    assert simulate(frame, symbol="TEST") == []


def test_simulate_time_stop_after_ten_held_days() -> None:
    trades = simulate(_declining_then_recovering_frame(), symbol="TEST")
    assert len(trades) == 1
    trade = trades[0]
    assert trade.held_days == 10
    assert trade.exit_reason == "time_stop"


def test_summarize_reports_expectancy_win_rate_and_drawdown() -> None:
    trades = [
        Rsi2Trade(
            symbol="TEST",
            entry_index=0,
            entry_fill_index=1,
            exit_index=2,
            exit_fill_index=3,
            entry_open=100.0,
            exit_open=101.0,
            held_days=2,
            exit_reason="sma5",
            ret_gross_pct=1.0,
            ret_net_base_pct=0.5,
            ret_net_stress_pct=0.4,
        ),
        Rsi2Trade(
            symbol="TEST",
            entry_index=5,
            entry_fill_index=6,
            exit_index=7,
            exit_fill_index=8,
            entry_open=100.0,
            exit_open=99.0,
            held_days=2,
            exit_reason="sma5",
            ret_gross_pct=-1.0,
            ret_net_base_pct=-0.5,
            ret_net_stress_pct=-0.6,
        ),
    ]
    summary = summarize(trades, BASE_COST_BPS)
    assert summary["trades"] == 2
    assert summary["expectancy_pct"] == pytest.approx(0.0)
    assert summary["win_rate"] == 0.5
    assert summary["profit_factor"] == pytest.approx(1.0)
    assert summary["max_drawdown_pct"] == pytest.approx(-0.5)


def test_random_baseline_reports_95th_percentile() -> None:
    frame = _daily_frame(np.linspace(50.0, 100.0, 300))
    result = random_baseline(frame, np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0]), BASE_COST_BPS)
    assert result["runs"] == 1000
    assert 0.0 <= result["baseline_pct"] <= 100.0
