from datetime import time

import pandas as pd
import pytest

from intraday_bot.bands import (
    bands_for_slot,
    is_decision_slot,
    prepare_session,
    session_vwap,
    sigma_by_slot,
    slot_end,
)

ET = "America/New_York"


def _frame(rows, date="2025-01-06"):
    index = pd.date_range(f"{date} 09:30", periods=len(rows), freq="5min", tz=ET)
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def test_sigma_uses_only_the_prior_sessions_and_bar_end_slots():
    prior = [
        _frame([(100, 101, 99, 101, 1000), (100, 102, 99, 99, 1000), (100, 103, 99, 102, 1000)]),
        _frame([(200, 201, 199, 198, 1000), (200, 203, 197, 202, 1000), (200, 205, 199, 204, 1000)]),
    ]
    sigma = sigma_by_slot(prior, bar_minutes=5, lookback=2)
    assert sigma[575] == pytest.approx(0.01)
    assert sigma[580] == pytest.approx(0.01)
    assert sigma[585] == pytest.approx(0.02)


def test_bands_are_gap_aware():
    bands = bands_for_slot(open_price=100.0, prior_close=102.0, sigma={600: 0.01}, slot=600, mult=1.0)
    assert bands is not None
    assert bands.anchor_high == 102.0
    assert bands.anchor_low == 100.0
    assert bands.upper == pytest.approx(103.02)
    assert bands.lower == pytest.approx(99.0)


def test_sigma_falls_back_to_an_earlier_slot():
    bands = bands_for_slot(open_price=100.0, prior_close=100.0, sigma={600: 0.01}, slot=630, mult=1.0)
    assert bands is not None
    assert bands.upper == pytest.approx(101.0)


def test_session_vwap_is_cumulative_and_causal():
    frame = _frame([(100, 100, 100, 100, 10), (100, 110, 100, 110, 10)])
    vwap = session_vwap(frame)
    assert vwap.iloc[0] == pytest.approx(100.0)
    typical_2 = (110 + 100 + 110) / 3
    assert vwap.iloc[1] == pytest.approx((100 * 10 + typical_2 * 10) / 20)


def test_prepare_session_marks_the_bar_end():
    frame = _frame([(100, 101, 99, 100, 10)])
    prepared = prepare_session(frame, bar_minutes=5)
    assert prepared["slot"].iloc[0] == 575  # 09:35
    assert slot_end(prepared.index[0], 5) == 575


def test_decision_slot_is_half_hourly():
    assert is_decision_slot(600, time(10, 0), time(15, 30), 30)
    assert not is_decision_slot(605, time(10, 0), time(15, 30), 30)
    assert is_decision_slot(630, time(10, 0), time(15, 30), 30)
    assert not is_decision_slot(945, time(10, 0), time(15, 30), 30)
