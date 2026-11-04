from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from research.london_sdt import (
    SdtConfig,
    _prior_day,
    _trend_side,
    clean_bars,
    load_bars,
    run_sdt,
    trail_stop,
    variant_config,
)


def _frame(rows, day="2026-01-06", start="04:00"):
    index = pd.date_range(f"{day} {start}", periods=len(rows), freq="5min", tz="UTC")
    values = np.array(rows, float)
    return pd.DataFrame(values, index=index, columns=["open", "high", "low", "close"])


def _prior():
    index = pd.date_range("2026-01-05 12:00", periods=2, freq="5min", tz="UTC")
    values = np.array(
        [
            (99.0, 100.0, 90.0, 99.0),
            (99.0, 99.5, 90.5, 99.0),
        ],
        float,
    )
    return pd.DataFrame(values, index=index, columns=["open", "high", "low", "close"])


def _bars(rows, day="2026-01-06", start="04:00"):
    return pd.concat([_prior(), _frame(rows, day=day, start=start)])


def test_long_breakout_fills_next_open_and_hits_two_r():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
            (101.8, 103.0, 101.0, 102.5),
            (102.5, 106.5, 102.0, 105.5),
        ]
    )
    result = run_sdt(bars, SdtConfig(cost_r=0.0))
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.entry == pytest.approx(101.2)
    assert trade.initial_stop == pytest.approx(98.8)
    assert trade.risk == pytest.approx(2.4)
    assert trade.target == pytest.approx(106.0)
    assert trade.exit == pytest.approx(106.0)
    assert trade.reason == "target"
    assert trade.gross_r == pytest.approx(2.0)
    assert trade.net_r == pytest.approx(2.0)
    assert trade.entry_time == "2026-01-06 04:05:00+00:00"
    assert trade.exit_time == "2026-01-06 04:20:00+00:00"
    assert trade.bars_held == 3
    assert trade.trailed is True


def test_trailing_stop_ratchets_up_and_never_loosens():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
            (101.8, 102.5, 100.9, 101.5),
            (101.5, 102.0, 100.5, 101.2),
        ]
    )
    trade = run_sdt(bars, SdtConfig(target_r=5.0, cost_r=0.0)).trades[0]
    assert trade.reason == "stop"
    assert trade.final_stop == pytest.approx(100.9)
    assert trade.exit == pytest.approx(100.9)
    assert trade.trailed is True


def test_short_breakout_trails_the_high_and_never_loosens():
    bars = _bars(
        [
            (90.5, 89.5, 89.0, 89.0),
            (88.8, 88.5, 88.0, 88.2),
            (88.2, 88.2, 87.8, 88.0),
            (88.0, 88.6, 87.9, 88.4),
        ]
    )
    trade = run_sdt(bars, SdtConfig(target_r=5.0, cost_r=0.0)).trades[0]
    assert trade.side == "short"
    assert trade.entry == pytest.approx(88.8)
    assert trade.initial_stop == pytest.approx(89.5)
    assert trade.final_stop == pytest.approx(88.2)
    assert trade.exit == pytest.approx(88.2)
    assert trade.reason == "stop"
    assert trade.trailed is True


def test_one_trade_per_day():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
            (101.8, 103.0, 101.0, 102.5),
            (102.5, 106.5, 102.0, 105.5),
            (105.5, 106.0, 105.0, 105.8),
            (105.8, 106.0, 105.2, 105.9),
        ]
    )
    result = run_sdt(bars, SdtConfig(cost_r=0.0))
    assert len(result.trades) == 1


def test_time_exit_at_session_close():
    config = SdtConfig(session_start=time(10, 0), session_end=time(10, 15), cost_r=0.0)
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 101.5, 101.0, 101.2),
            (101.2, 101.4, 101.1, 101.3),
        ],
        start="10:00",
    )
    trade = run_sdt(bars, config).trades[0]
    assert trade.reason == "time"
    assert trade.exit == pytest.approx(101.3)
    assert trade.bars_held == 2


def test_cost_is_deducted_from_r():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
            (101.8, 103.0, 101.0, 102.5),
            (102.5, 106.5, 102.0, 105.5),
        ]
    )
    trade = run_sdt(bars, SdtConfig(cost_r=0.05)).trades[0]
    assert trade.gross_r == pytest.approx(2.0)
    assert trade.net_r == pytest.approx(1.95)


def test_same_bar_stop_and_target_uses_stop_first():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 106.5, 98.0, 101.0),
        ]
    )
    trade = run_sdt(bars, SdtConfig(cost_r=0.0)).trades[0]
    assert trade.reason == "stop"
    assert trade.exit == pytest.approx(98.8)
    assert trade.gross_r == pytest.approx(-1.0)


def test_no_trade_without_a_prior_day():
    bars = _frame(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
        ]
    )
    assert run_sdt(bars, SdtConfig()).trades == []


def test_no_entry_on_the_last_session_bar():
    config = SdtConfig(session_start=time(10, 50), session_end=time(11, 0))
    bars = _bars(
        [
            (99.0, 100.0, 98.0, 99.0),
            (99.0, 101.5, 98.8, 101.0),
        ],
        start="10:50",
    )
    assert run_sdt(bars, config).trades == []


def test_trail_stop_is_a_one_way_ratchet():
    assert trail_stop("long", 100.0, 99.5, 101.0) == pytest.approx(100.0)
    assert trail_stop("long", 100.0, 100.5, 101.0) == pytest.approx(100.5)
    assert trail_stop("short", 100.0, 99.0, 100.5) == pytest.approx(100.0)
    assert trail_stop("short", 100.0, 99.0, 99.5) == pytest.approx(99.5)


def test_load_bars_reads_a_datetime_column_as_utc(tmp_path):
    target = tmp_path / "gold.csv"
    target.write_text(
        "datetime,open,high,low,close,volume\n"
        "2026-01-05 12:00:00,99,100,90,99,10\n"
        "2026-01-05 12:05:00,99,99.5,90.5,99,10\n",
        encoding="utf-8",
    )
    bars = load_bars(target)
    assert len(bars) == 2
    assert str(bars.index.tz) == "UTC"
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


def test_load_bars_reads_epoch_milliseconds_as_utc(tmp_path):
    target = tmp_path / "gold_ms.csv"
    target.write_text(
        "timestamp,open,high,low,close\n"
        "1672617600000,1826.637,1826.637,1826.637,1826.637\n"
        "1672617900000,1826.637,1826.637,1826.637,1826.637\n",
        encoding="utf-8",
    )
    bars = load_bars(target)
    assert len(bars) == 2
    assert bars.index[0] == pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    assert bars.index[1] == pd.Timestamp("2023-01-02 00:05:00", tz="UTC")
    assert str(bars.index.tz) == "UTC"


def test_load_bars_reads_epoch_seconds_as_utc(tmp_path):
    target = tmp_path / "gold_s.csv"
    target.write_text(
        "timestamp,open,high,low,close\n"
        "1672617600,1826.637,1826.637,1826.637,1826.637\n"
        "1672617900,1826.637,1826.637,1826.637,1826.637\n",
        encoding="utf-8",
    )
    bars = load_bars(target)
    assert len(bars) == 2
    assert bars.index[0] == pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    assert bars.index[1] == pd.Timestamp("2023-01-02 00:05:00", tz="UTC")


def test_clean_bars_drops_flat_padding():
    index = pd.to_datetime(
        [
            "2026-01-05 09:00:00",
            "2026-01-05 09:05:00",
            "2026-01-05 09:10:00",
        ],
        utc=True,
    )
    bars = pd.DataFrame(
        [
            (100.0, 101.0, 99.0, 100.5),
            (100.5, 100.5, 100.5, 100.5),
            (100.5, 102.0, 100.0, 101.0),
        ],
        index=index,
        columns=["open", "high", "low", "close"],
    )
    cleaned = clean_bars(bars)
    assert list(cleaned.index) == [index[0], index[2]]


def test_clean_bars_drops_closed_market_bars():
    index = pd.to_datetime(
        [
            "2026-01-02 21:55:00",
            "2026-01-02 22:00:00",
            "2026-01-03 10:00:00",
            "2026-01-04 21:55:00",
            "2026-01-04 22:00:00",
            "2026-01-05 00:00:00",
        ],
        utc=True,
    )
    bars = pd.DataFrame(
        [(100.0, 101.0, 99.0, 100.5)] * len(index),
        index=index,
        columns=["open", "high", "low", "close"],
    )
    cleaned = clean_bars(bars)
    assert list(cleaned.index) == [index[0], index[4], index[5]]


def test_prior_day_skips_sunday_and_uses_friday():
    friday = _frame([(99.0, 100.0, 90.0, 99.0)], day="2026-01-02", start="12:00")
    sunday = _frame([(105.0, 110.0, 80.0, 105.0)], day="2026-01-04", start="22:00")
    monday = _frame([(100.0, 101.0, 99.0, 100.5)], day="2026-01-05", start="04:00")
    daily = {
        friday.index[0].date(): friday,
        sunday.index[0].date(): sunday,
        monday.index[0].date(): monday,
    }
    prior = _prior_day(daily, date(2026, 1, 5), 7)
    assert prior is friday


def test_run_sdt_uses_friday_not_sunday_for_monday():
    friday = _frame(
        [(99.0, 100.0, 90.0, 99.0), (99.0, 100.0, 90.0, 99.0)],
        day="2026-01-02",
        start="12:00",
    )
    sunday = _frame(
        [(105.0, 110.0, 80.0, 105.0), (105.0, 110.0, 80.0, 105.0)],
        day="2026-01-04",
        start="22:00",
    )
    monday = _frame(
        [(100.5, 101.5, 99.0, 101.0), (101.2, 102.0, 100.0, 101.5)],
        day="2026-01-05",
        start="04:00",
    )
    result = run_sdt(pd.concat([friday, sunday, monday]), SdtConfig(cost_r=0.0))
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.day == "2026-01-05"
    assert trade.side == "long"
    assert trade.entry == pytest.approx(101.2)


def test_default_config_is_baseline_and_variants_are_isolated():
    baseline = SdtConfig()
    assert baseline.range_filter is False
    assert baseline.trend_filter is False
    assert baseline.trail_mode == "always"
    assert variant_config("A") == baseline

    b = variant_config("B")
    assert b.range_filter is True and b.trend_filter is False
    assert b.trail_mode == "always"

    c = variant_config("C")
    assert c.trend_filter is True and c.range_filter is False
    assert c.trail_mode == "always"

    assert variant_config("D").trail_mode == "never"
    assert variant_config("E").trail_mode == "after_1r"


def test_range_filter_skips_wide_prior_day():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
            (101.8, 103.0, 101.0, 102.5),
            (102.5, 106.5, 102.0, 105.5),
        ]
    )
    assert len(run_sdt(bars, SdtConfig(cost_r=0.0)).trades) == 1
    assert run_sdt(bars, SdtConfig(range_filter=True, cost_r=0.0)).trades == []


def test_range_filter_allows_inside_prior_day():
    prior = pd.DataFrame(
        [
            (99.5, 100.0, 99.5, 99.8),
            (99.8, 100.0, 99.6, 99.9),
        ],
        index=pd.date_range("2026-01-05 12:00", periods=2, freq="5min", tz="UTC"),
        columns=["open", "high", "low", "close"],
    )
    current = _frame(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
        ]
    )
    result = run_sdt(
        pd.concat([prior, current]),
        SdtConfig(range_filter=True, cost_r=0.0),
    )
    assert len(result.trades) == 1


def test_trend_filter_uses_20_trading_day_close():
    dates = pd.bdate_range("2026-01-05", periods=22, tz="UTC")
    rising = {
        day.date(): pd.DataFrame(
            [(close - 0.5, close + 0.5, close - 1.0, close)],
            index=pd.DatetimeIndex([day + pd.Timedelta(hours=12)]),
            columns=["open", "high", "low", "close"],
        )
        for day, close in zip(dates, [100.0 + i for i in range(22)])
    }
    assert _trend_side(rising, dates[-1].date(), 20) == "long"
    falling = {
        day.date(): pd.DataFrame(
            [(close - 0.5, close + 0.5, close - 1.0, close)],
            index=pd.DatetimeIndex([day + pd.Timedelta(hours=12)]),
            columns=["open", "high", "low", "close"],
        )
        for day, close in zip(dates, [100.0 - i for i in range(22)])
    }
    assert _trend_side(falling, dates[-1].date(), 20) == "short"
    assert _trend_side(rising, dates[19].date(), 20) is None


def test_trend_filter_blocks_opposite_side():
    dates = pd.bdate_range("2026-01-05", periods=23, tz="UTC")
    frames = []
    for day, close in zip(dates[:-1], [100.0 + i for i in range(22)]):
        frames.append(
            pd.DataFrame(
                [(close - 0.5, close + 0.5, close - 1.0, close)],
                index=pd.DatetimeIndex([day + pd.Timedelta(hours=12)]),
                columns=["open", "high", "low", "close"],
            )
        )
    current_day = dates[-1]
    current = pd.DataFrame(
        [
            (98.0, 99.0, 97.0, 98.5),
            (98.6, 99.0, 98.0, 98.8),
        ],
        index=pd.DatetimeIndex(
            [
                current_day + pd.Timedelta(hours=4),
                current_day + pd.Timedelta(hours=4, minutes=5),
            ]
        ),
        columns=["open", "high", "low", "close"],
    )
    bars = pd.concat(frames + [current])
    assert run_sdt(bars, SdtConfig(trend_filter=True, cost_r=0.0)).trades == []


def test_no_trailing_variant_keeps_fixed_stop():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
            (101.8, 102.5, 100.9, 101.5),
            (101.5, 102.0, 100.5, 101.2),
        ]
    )
    trade = run_sdt(bars, SdtConfig(trail_mode="never", cost_r=0.0)).trades[0]
    assert trade.reason == "data_end"
    assert trade.final_stop == pytest.approx(98.8)
    assert trade.trailed is False


def test_trailing_after_one_r_waits_for_one_r():
    bars = _bars(
        [
            (99.0, 101.5, 98.8, 101.0),
            (101.2, 102.0, 100.8, 101.8),
            (101.8, 102.2, 100.5, 101.5),
            (101.5, 103.8, 102.0, 103.0),
            (103.0, 103.2, 101.5, 101.8),
        ]
    )
    baseline = run_sdt(bars, SdtConfig(cost_r=0.0)).trades[0]
    after = run_sdt(bars, SdtConfig(trail_mode="after_1r", cost_r=0.0)).trades[0]
    assert baseline.reason == "stop"
    assert baseline.exit == pytest.approx(100.8)
    assert after.reason == "stop"
    assert after.exit == pytest.approx(102.0)
    assert after.final_stop == pytest.approx(102.0)
    assert after.trailed is True
