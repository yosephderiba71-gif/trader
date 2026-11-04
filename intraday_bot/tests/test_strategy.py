from datetime import datetime

import pytest

from intraday_bot.bands import Bands
from intraday_bot.config import IntradaySettings
from intraday_bot.strategy import Position, decide, size_position

ET = "America/New_York"


def _now(hour=10, minute=0, day=6):
    from zoneinfo import ZoneInfo

    return datetime(2025, 1, day, hour, minute, tzinfo=ZoneInfo(ET))


def _bands():
    return Bands(upper=101.0, lower=99.0, sigma=0.01, anchor_high=101.0, anchor_low=99.0)


def test_long_entry_above_upper_at_a_decision_time():
    s = IntradaySettings()
    d = decide(_now(), 101.5, _bands(), 100.5, None, 0, 100_000, s)
    assert d.action == "enter_long"
    assert d.reason == "band_break"
    assert d.stop == pytest.approx(101.0)
    assert d.qty == 500  # $250 risk / $0.50 per share, under the notional cap


def test_no_entry_inside_the_band():
    s = IntradaySettings()
    d = decide(_now(), 100.5, _bands(), 100.0, None, 0, 100_000, s)
    assert d.action == "flat"
    assert d.reason == "inside_band"


def test_long_only_ignores_a_lower_break():
    s = IntradaySettings()
    d = decide(_now(), 98.5, _bands(), 99.5, None, 0, 100_000, s)
    assert d.action == "flat"
    assert d.reason == "short_signal_ignored"


def test_one_trade_per_day():
    s = IntradaySettings()
    d = decide(_now(), 101.5, _bands(), 100.5, None, 1, 100_000, s)
    assert d.action == "flat"
    assert d.reason == "max_trades"


def test_no_entry_before_the_window():
    s = IntradaySettings()
    d = decide(_now(9, 45), 101.5, _bands(), 100.5, None, 0, 100_000, s)
    assert d.action == "flat"
    assert d.reason == "before_window"


def test_no_entry_between_decision_points():
    s = IntradaySettings()
    d = decide(_now(10, 5), 101.5, _bands(), 100.5, None, 0, 100_000, s)
    assert d.action == "flat"
    assert d.reason == "not_decision_time"


def test_trailing_stop_is_the_higher_of_band_and_vwap():
    s = IntradaySettings()
    pos = Position(100, 101.5, _now(), 101.0, 101.0, 100.5)
    d = decide(_now(10, 30), 100.9, _bands(), 100.8, pos, 1, 100_000, s)
    assert d.action == "exit_long"
    assert d.reason == "trail_stop"
    assert d.stop == pytest.approx(101.0)

    d2 = decide(_now(10, 30), 101.2, _bands(), 100.8, pos, 1, 100_000, s)
    assert d2.action == "hold"
    assert d2.stop == pytest.approx(101.0)


def test_flat_by_close_wins_over_everything():
    s = IntradaySettings()
    pos = Position(100, 101.5, _now(), 101.0, 101.0, 100.5)
    d = decide(_now(15, 45), 105.0, _bands(), 104.0, pos, 1, 100_000, s)
    assert d.action == "exit_long"
    assert d.reason == "flat_by_close"


def test_size_respects_risk_and_notional_cap():
    s = IntradaySettings()
    assert size_position(100_000, 101.5, 101.0, s) == (500, 250.0)
    qty, risk = size_position(100_000, 101.5, 101.4, s)
    assert qty == 985
    assert risk == pytest.approx(98.5)
    assert size_position(100, 101.5, 101.0, s) is None


def test_stop_is_not_checked_between_decision_points():
    s = IntradaySettings()
    pos = Position(100, 101.5, _now(), 101.0, 101.0, 100.5)
    d = decide(_now(10, 5), 100.5, _bands(), 100.8, pos, 1, 100_000, s)
    assert d.action == "hold"
    assert d.reason == "between_checks"
    assert d.stop == pytest.approx(101.0)
