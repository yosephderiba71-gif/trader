from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from research.london_range_breakout import range_variant_config, run_range_variant


def _frame(rows):
    index = pd.to_datetime([row[0] for row in rows], utc=True)
    values = [row[1:] for row in rows]
    return pd.DataFrame(values, index=index, columns=["open", "high", "low", "close"])


def _range_rows():
    return [
        ("2026-01-06 00:00", 99.0, 100.0, 99.0, 99.5),
        ("2026-01-06 06:55", 99.5, 100.0, 99.0, 99.8),
    ]


def test_variant_configs_are_fixed():
    f = range_variant_config("F")
    assert f.range_start == time(0, 0)
    assert f.range_end == time(7, 0)
    assert f.entry_start == time(7, 0)
    assert f.entry_end == time(11, 0)
    assert f.mode == "breakout"
    assert f.target_kind == "r"

    g = range_variant_config("G")
    assert g.range_start == time(7, 0)
    assert g.range_end == time(8, 0)
    assert g.entry_start == time(8, 0)
    assert g.entry_end == time(11, 0)

    h = range_variant_config("H")
    assert h.range_start == time(0, 0)
    assert h.range_end == time(7, 0)
    assert h.entry_start == time(7, 0)
    assert h.entry_end == time(11, 0)
    assert h.mode == "failed_break"
    assert h.target_kind == "midpoint"

    with pytest.raises(ValueError):
        range_variant_config("A")


def test_f_long_breakout_uses_range_stop_and_two_r_target():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 99.8, 100.6, 99.7, 100.5),
            ("2026-01-06 07:05", 100.6, 104.0, 100.5, 103.5),
        ]
    )
    trade = run_range_variant(bars, range_variant_config("F", cost_r=0.0)).trades[0]
    assert trade.variant == "F"
    assert trade.side == "long"
    assert trade.entry == pytest.approx(100.6)
    assert trade.initial_stop == pytest.approx(99.0)
    assert trade.risk == pytest.approx(1.6)
    assert trade.target == pytest.approx(103.8)
    assert trade.exit == pytest.approx(103.8)
    assert trade.reason == "target"
    assert trade.net_r == pytest.approx(2.0)


def test_f_short_breakout_uses_range_stop_and_two_r_target():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 99.5, 99.6, 98.4, 98.5),
            ("2026-01-06 07:05", 98.4, 98.6, 95.0, 95.5),
        ]
    )
    trade = run_range_variant(bars, range_variant_config("F", cost_r=0.0)).trades[0]
    assert trade.side == "short"
    assert trade.entry == pytest.approx(98.4)
    assert trade.initial_stop == pytest.approx(100.0)
    assert trade.target == pytest.approx(95.2)
    assert trade.exit == pytest.approx(95.2)
    assert trade.reason == "target"


def test_g_uses_0700_to_0800_range():
    bars = _frame(
        [
            ("2026-01-06 07:00", 99.0, 100.0, 99.0, 99.5),
            ("2026-01-06 07:55", 99.5, 100.0, 99.0, 99.8),
            ("2026-01-06 08:00", 99.8, 100.6, 99.7, 100.5),
            ("2026-01-06 08:05", 100.6, 104.0, 100.5, 103.5),
        ]
    )
    trade = run_range_variant(bars, range_variant_config("G", cost_r=0.0)).trades[0]
    assert trade.variant == "G"
    assert trade.range_high == pytest.approx(100.0)
    assert trade.range_low == pytest.approx(99.0)
    assert trade.entry == pytest.approx(100.6)
    assert trade.target == pytest.approx(103.8)
    assert trade.reason == "target"


def test_h_fades_failed_downside_break_to_midpoint():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 99.2, 99.3, 98.2, 98.5),
            ("2026-01-06 07:05", 98.6, 99.6, 98.5, 99.4),
            ("2026-01-06 07:10", 99.2, 99.8, 99.0, 99.6),
        ]
    )
    trade = run_range_variant(bars, range_variant_config("H", cost_r=0.0)).trades[0]
    assert trade.variant == "H"
    assert trade.side == "long"
    assert trade.entry == pytest.approx(99.2)
    assert trade.initial_stop == pytest.approx(98.2)
    assert trade.target == pytest.approx(99.5)
    assert trade.exit == pytest.approx(99.5)
    assert trade.reason == "target"


def test_h_fades_failed_upside_break_to_midpoint():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 100.5, 100.8, 100.0, 100.6),
            ("2026-01-06 07:05", 100.2, 100.4, 99.4, 99.6),
            ("2026-01-06 07:10", 99.8, 100.0, 99.2, 99.3),
        ]
    )
    trade = run_range_variant(bars, range_variant_config("H", cost_r=0.0)).trades[0]
    assert trade.side == "short"
    assert trade.entry == pytest.approx(99.8)
    assert trade.initial_stop == pytest.approx(100.8)
    assert trade.target == pytest.approx(99.5)
    assert trade.exit == pytest.approx(99.5)
    assert trade.reason == "target"


def test_h_does_not_trade_when_reentry_is_after_three_candles():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 99.2, 99.3, 98.2, 98.5),
            ("2026-01-06 07:05", 98.5, 98.6, 98.1, 98.4),
            ("2026-01-06 07:10", 98.4, 98.5, 98.0, 98.3),
            ("2026-01-06 07:15", 98.3, 98.4, 97.9, 98.2),
            ("2026-01-06 07:20", 98.6, 99.6, 98.5, 99.4),
        ]
    )
    assert run_range_variant(bars, range_variant_config("H", cost_r=0.0)).trades == []


def test_time_exit_at_1100_utc():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 99.8, 100.6, 99.7, 100.5),
            ("2026-01-06 07:05", 100.6, 101.0, 100.5, 100.8),
            ("2026-01-06 10:55", 100.8, 101.0, 100.7, 100.9),
        ]
    )
    trade = run_range_variant(bars, range_variant_config("F", cost_r=0.0)).trades[0]
    assert trade.reason == "time"
    assert trade.exit == pytest.approx(100.9)


def test_one_trade_per_day():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 99.8, 100.6, 99.7, 100.5),
            ("2026-01-06 07:05", 100.6, 104.0, 100.5, 103.5),
            ("2026-01-06 07:10", 103.5, 104.0, 103.0, 103.8),
            ("2026-01-06 07:15", 103.8, 104.2, 103.0, 103.5),
        ]
    )
    result = run_range_variant(bars, range_variant_config("F", cost_r=0.0))
    assert len(result.trades) == 1


def test_cost_is_deducted_from_r():
    bars = _frame(
        _range_rows()
        + [
            ("2026-01-06 07:00", 99.8, 100.6, 99.7, 100.5),
            ("2026-01-06 07:05", 100.6, 104.0, 100.5, 103.5),
        ]
    )
    trade = run_range_variant(bars, range_variant_config("F", cost_r=0.05)).trades[0]
    assert trade.gross_r == pytest.approx(2.0)
    assert trade.net_r == pytest.approx(1.95)
