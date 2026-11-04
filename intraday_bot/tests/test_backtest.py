from datetime import time

import pandas as pd
import pytest

from intraday_bot.backtest import run_backtest
from intraday_bot.config import IntradaySettings

ET = "America/New_York"


def _rows(closes, opens=None, volume=1000):
    opens = opens or closes
    rows = []
    for o, c in zip(opens, closes):
        rows.append((o, max(o, c) + 0.2, min(o, c) - 0.2, c, volume))
    return rows


def _frame(rows, date):
    index = pd.date_range(f"{date} 09:30", periods=len(rows), freq="5min", tz=ET)
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def _bars(today_close=102.0):
    prior = _rows([100, 100, 100, 100, 100, 101, 100, 100, 100, 100, 100, 100, 100])
    today = _rows(
        [100, 100, 100, 100, 100, today_close, 103, 103.5, 103.5, 103.5, 103.5, 104, 104],
        opens=[100, 100, 100, 100, 100, 102, 102.5, 103.5, 103.5, 103.5, 103.5, 104, 104],
    )
    return pd.concat(
        [
            _frame(prior, "2025-01-06"),
            _frame(today, "2025-01-07"),
        ]
    )


def _settings(**overrides):
    base = dict(
        band_lookback_sessions=1,
        entry_start=time(10, 0),
        entry_end=time(10, 0),
        flat_time=time(10, 30),
        decision_minutes=30,
        risk_per_trade=0.01,
        max_position_pct=1.0,
        slippage_bps=0.0,
        commission_per_share=0.0,
    )
    base.update(overrides)
    return IntradaySettings(**base)


def test_backtest_enters_next_open_and_flats_at_the_close():
    result = run_backtest(_bars(), _settings(), starting_equity=100_000)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(102.5)
    assert trade["exit_price"] == pytest.approx(104.0)
    assert trade["reason"] == "flat_by_close"
    assert trade["qty"] == 666  # $1,000 risk / $1.50 per share, under the cap
    assert trade["pnl"] == pytest.approx((104.0 - 102.5) * 666)


def test_costs_are_charged_on_both_sides():
    result = run_backtest(_bars(), _settings(slippage_bps=10, commission_per_share=0.01), starting_equity=100_000)
    trade = result.trades.iloc[0]
    assert trade["entry_price"] > 102.5
    assert trade["exit_price"] < 104.0
    assert trade["pnl"] < (104.0 - 102.5) * 666


def test_no_trade_when_price_stays_inside_the_band():
    result = run_backtest(_bars(today_close=100.5), _settings(), starting_equity=100_000)
    assert result.trades.empty


def test_metrics_report_the_trade():
    result = run_backtest(_bars(), _settings(), starting_equity=100_000)
    assert result.metrics["trades"] == 1
    assert result.metrics["wins"] == 1
    assert result.metrics["win_rate"] == pytest.approx(1.0)
    assert result.metrics["total_return"] > 0
    assert result.metrics["max_drawdown"] <= 0
