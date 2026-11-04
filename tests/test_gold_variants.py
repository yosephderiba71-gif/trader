from __future__ import annotations

import pandas as pd
import pytest

from research.gold_variants import (
    DailyJConfig,
    DailyKConfig,
    ReversionConfig,
    add_indicators,
    adx_wilder,
    atr_wilder,
    rsi_wilder,
    run_daily_j,
    run_daily_k,
    run_reversion,
    summarize_reversion,
)


def _intraday_frame(closes, start="2026-01-06 00:00"):
    index = pd.date_range(start, periods=len(closes), freq="5min", tz="UTC")
    opens = [closes[0]] + list(closes[:-1])
    highs = [max(open_, close) + 0.1 for open_, close in zip(opens, closes)]
    lows = [min(open_, close) - 0.1 for open_, close in zip(opens, closes)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )


def _long_signal_bars():
    closes = [100.0 - 0.2 * i for i in range(20)] + [92.0, 92.5, 94.0, 95.0, 96.0, 96.5, 97.0]
    return _intraday_frame(closes)


def _short_signal_bars():
    closes = [100.0 + 0.2 * i for i in range(20)] + [108.0, 107.5, 106.0, 105.0, 104.0, 103.5, 103.0]
    return _intraday_frame(closes)


def _daily_frame(closes, opens=None):
    index = pd.bdate_range("2025-01-01", periods=len(closes))
    if opens is None:
        opens = [close - 0.2 for close in closes]
    highs = [max(open_, close) + 0.5 for open_, close in zip(opens, closes)]
    lows = [min(open_, close) - 0.5 for open_, close in zip(opens, closes)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )


def test_rsi_wilder_handles_one_sided_and_flat_series():
    rising = pd.Series([100.0 + i for i in range(20)])
    falling = pd.Series([100.0 - i for i in range(20)])
    flat = pd.Series([100.0] * 20)
    assert rsi_wilder(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi_wilder(falling, 14).iloc[-1] == pytest.approx(0.0)
    assert rsi_wilder(flat, 14).iloc[-1] == pytest.approx(50.0)


def test_atr_wilder_and_adx_wilder_on_constant_range():
    close = pd.Series([100.0 + i * 0.5 for i in range(30)])
    high = close + 0.5
    low = close - 0.5
    atr = atr_wilder(high, low, close, 14)
    adx = adx_wilder(high, low, close, 14)
    assert atr.iloc[-1] == pytest.approx(1.0)
    assert adx.iloc[-1] == pytest.approx(100.0)


def test_bollinger_bands_use_population_std():
    bars = _intraday_frame([100.0] * 25)
    indicators = add_indicators(bars)
    assert indicators["bb_upper"].iloc[-1] == pytest.approx(100.0)
    assert indicators["bb_lower"].iloc[-1] == pytest.approx(100.0)


def test_i_long_enters_next_open_and_exits_at_sma_touch():
    bars = _long_signal_bars()
    config = ReversionConfig(cost_r=0.0)
    trade = run_reversion(bars, config, clean=False).trades[0]
    indicators = add_indicators(bars, config)
    assert trade.side == "long"
    assert trade.entry == pytest.approx(92.0)
    assert trade.entry_time == "2026-01-06 01:45:00+00:00"
    assert trade.reason == "sma"
    assert trade.exit == pytest.approx(indicators["exit_sma"].iloc[25])
    assert trade.bars_held == 5


def test_i_short_enters_next_open_and_exits_at_sma_touch():
    bars = _short_signal_bars()
    config = ReversionConfig(cost_r=0.0)
    trade = run_reversion(bars, config, clean=False).trades[0]
    indicators = add_indicators(bars, config)
    assert trade.side == "short"
    assert trade.entry == pytest.approx(108.0)
    assert trade.reason == "sma"
    assert trade.exit == pytest.approx(indicators["exit_sma"].iloc[25])


def test_i_stop_is_two_atr_and_cost_is_deducted():
    bars = _long_signal_bars()
    rows = bars.copy()
    rows.iloc[21, rows.columns.get_loc("open")] = 92.0
    rows.iloc[21, rows.columns.get_loc("high")] = 92.6
    rows.iloc[21, rows.columns.get_loc("low")] = 90.0
    rows.iloc[21, rows.columns.get_loc("close")] = 90.5
    config = ReversionConfig(cost_r=0.05)
    trade = run_reversion(rows, config, clean=False).trades[0]
    assert trade.risk == pytest.approx(2.0 * add_indicators(rows, config)["atr"].iloc[20])
    assert trade.initial_stop == pytest.approx(trade.entry - trade.risk)
    assert trade.reason == "stop"
    assert trade.gross_r == pytest.approx(-1.0)
    assert trade.net_r == pytest.approx(-1.05)


def test_i_time_exit_uses_configured_bar_limit():
    bars = _long_signal_bars()
    config = ReversionConfig(max_bars=1, cost_r=0.0)
    trade = run_reversion(bars, config, clean=False).trades[0]
    assert trade.reason == "time"
    assert trade.bars_held == 1
    assert trade.exit == pytest.approx(bars["close"].iloc[21])


def test_i_reports_worst_loss_and_average_win_loss():
    bars = _long_signal_bars()
    rows = bars.copy()
    rows.iloc[21, rows.columns.get_loc("open")] = 92.0
    rows.iloc[21, rows.columns.get_loc("high")] = 92.6
    rows.iloc[21, rows.columns.get_loc("low")] = 90.0
    rows.iloc[21, rows.columns.get_loc("close")] = 90.5
    trade = run_reversion(rows, ReversionConfig(cost_r=0.05), clean=False).trades[0]
    metrics = summarize_reversion([trade])
    assert metrics["trades"] == 1
    assert metrics["win_rate"] == pytest.approx(0.0)
    assert metrics["avg_loss_r"] == pytest.approx(-1.05)
    assert metrics["worst_loss_r"] == pytest.approx(-1.05)


def test_j_long_enters_next_open_and_exits_when_condition_breaks():
    closes = [100.0 + i * 0.5 for i in range(130)]
    opens = [close - 0.2 for close in closes]
    for i in range(115, len(closes)):
        opens[i] = closes[i] + 0.5
    bars = _daily_frame(closes, opens)
    result = run_daily_j(bars, DailyJConfig())
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.entry_date == str(bars.index[110].date())
    assert trade.exit_date == str(bars.index[116].date())
    assert trade.entry == pytest.approx(opens[110])
    assert trade.exit == pytest.approx(opens[116])
    assert trade.reason == "condition"


def test_j_short_is_symmetric():
    closes = [200.0 - i * 0.5 for i in range(130)]
    opens = [close + 0.2 for close in closes]
    for i in range(115, len(closes)):
        opens[i] = closes[i] - 0.5
    bars = _daily_frame(closes, opens)
    result = run_daily_j(bars, DailyJConfig())
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short"
    assert trade.entry_date == str(bars.index[110].date())
    assert trade.exit_date == str(bars.index[116].date())
    assert trade.reason == "condition"
    assert trade.return_pct > 0.0


def test_j_waits_for_sma_history():
    closes = [100.0 + i * 0.5 for i in range(60)]
    bars = _daily_frame(closes)
    result = run_daily_j(bars, DailyJConfig())
    assert result.trades == []


def test_k_is_long_when_close_is_above_sma250_and_matches_buy_hold():
    closes = [100.0 + i * 0.5 for i in range(270)]
    bars = _daily_frame(closes)
    result = run_daily_k(bars, DailyKConfig())
    assert result.metrics["entries"] == 1
    assert result.metrics["days_invested"] == 20
    expected = (closes[-1] / closes[249] - 1.0) * 100.0
    assert result.metrics["strategy_return_pct"] == pytest.approx(expected)
    assert result.metrics["buy_hold_return_pct"] == pytest.approx(expected)
    assert result.yearly[str(bars.index[250].year)]["entries"] == 1


def test_k_stays_in_cash_below_sma250():
    closes = [200.0 - i * 0.5 for i in range(270)]
    bars = _daily_frame(closes)
    result = run_daily_k(bars, DailyKConfig())
    assert result.metrics["entries"] == 0
    assert result.metrics["days_invested"] == 0
    assert result.metrics["strategy_return_pct"] == pytest.approx(0.0)
    assert result.metrics["buy_hold_return_pct"] < 0.0
