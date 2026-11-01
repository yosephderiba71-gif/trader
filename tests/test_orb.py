import numpy as np
import pandas as pd
import pytest

from trader.orb import OrbConfig, run_orb, summarize, OrbTrade

CFG = OrbConfig(cost_r=0.0)


def _day(rows, date="2026-01-05"):
    """rows: list of (open, high, low, close) starting at 09:30 New York time, 5-minute bars."""
    idx = pd.date_range(f"{date} 09:30", periods=len(rows), freq="5min", tz="America/New_York")
    arr = np.array(rows, float)
    return pd.DataFrame(arr, index=idx, columns=["open", "high", "low", "close"])


def _inside(n):
    """Bars that stay inside the opening range 99-101."""
    return [(100.0, 101.0, 99.0, 100.0)] * n


def test_long_breakout_hits_target_and_fills_on_next_open():
    rows = _inside(6)                                # bars 0-5: window, range 99-101
    rows.append((101.0, 102.5, 100.8, 102.0))        # bar 6 closes above 101: signal
    rows.append((103.0, 109.0, 102.5, 108.5))        # bar 7 fills at 103 (next open)
    trades = run_orb(_day(rows), CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "long"
    assert t.entry == 103.0                          # not 102: the signal bar's close is not tradable
    # stop 99, risk 4, target 111: high 109 misses the target and low 102.5 misses the stop,
    # so the trade runs to the last bar and exits at its close, 108.5
    assert t.reason == "time"
    assert t.exit == pytest.approx(108.5)
    assert t.r == pytest.approx((108.5 - 103.0) / 4.0)


def test_target_exit_gives_target_multiple_in_r():
    rows = _inside(6)
    rows.append((101.0, 102.5, 100.8, 102.0))        # signal, range high 101
    rows.append((102.0, 108.0, 101.5, 107.0))        # fill 102, stop 99, risk 3, target 108
    t = run_orb(_day(rows), CFG)[0]
    assert t.reason == "target"
    assert t.exit == pytest.approx(108.0)
    assert t.r == pytest.approx(2.0)


def test_same_bar_touching_stop_and_target_counts_the_stop_first():
    rows = _inside(6)
    rows.append((101.0, 102.5, 100.8, 102.0))        # signal
    rows.append((102.0, 110.0, 98.0, 102.0))         # fill 102; low 98 hits stop 99, high 110 hits target 108
    t = run_orb(_day(rows), CFG)[0]
    assert t.reason == "stop"
    assert t.r == pytest.approx(-1.0)


def test_time_exit_after_max_hold():
    cfg = OrbConfig(cost_r=0.0, max_hold_bars=2)     # entry on bar 7, so the time exit is on bar 8
    rows = _inside(6)
    rows.append((101.0, 102.5, 100.8, 102.0))        # signal
    # Each later low stays above the trailed stop, so only the time exit can fire.
    rows += [(102.0, 103.0, 101.9, 102.5), (102.5, 103.0, 102.0, 102.7), (102.7, 103.0, 102.1, 102.9)]
    t = run_orb(_day(rows), cfg)[0]
    assert t.reason == "time"
    assert t.exit == pytest.approx(102.7)            # closes on bar 8, not on the last bar


def test_trailing_stop_follows_prior_low():
    cfg = OrbConfig(cost_r=0.0, max_hold_bars=6)
    rows = _inside(6)
    rows.append((101.0, 102.5, 100.8, 102.0))        # signal
    rows.append((102.0, 103.0, 101.9, 102.5))        # bar 7: low 101.9 becomes the stop
    rows.append((102.5, 103.0, 101.8, 102.5))        # bar 8: low 101.8 is below the trailed stop
    t = run_orb(_day(rows), cfg)[0]
    assert t.reason == "stop"
    assert t.exit == pytest.approx(101.9)
    assert t.r == pytest.approx(-0.1 / 3)


def test_one_trade_per_session():
    rows = _inside(6)
    rows.append((101.0, 102.5, 100.8, 102.0))        # first signal
    rows.append((102.0, 102.5, 98.0, 98.5))          # stop hit
    rows.append((98.5, 99.0, 97.0, 97.5))           # would be a short signal
    rows.append((97.5, 97.8, 96.0, 96.5))
    trades = run_orb(_day(rows), CFG)
    assert len(trades) == 1


def test_short_side_is_mirrored():
    rows = _inside(6)
    rows.append((99.0, 99.5, 97.5, 98.0))            # closes below 99: short signal
    rows.append((98.0, 98.5, 92.0, 92.5))            # fill 98, stop 101, risk 3, target 92
    t = run_orb(_day(rows), CFG)[0]
    assert t.side == "short"
    assert t.reason == "target"
    assert t.r == pytest.approx(2.0)


def test_no_trade_when_price_stays_inside_range():
    assert run_orb(_day(_inside(20)), CFG) == []


def test_costs_come_off_every_trade():
    rows = _inside(6)
    rows.append((101.0, 102.5, 100.8, 102.0))
    rows.append((102.0, 108.0, 101.5, 107.0))
    t = run_orb(_day(rows), OrbConfig(cost_r=0.05))[0]
    assert t.r == pytest.approx(1.95)


def test_rejects_bars_without_ohlc():
    with pytest.raises(ValueError):
        run_orb(pd.DataFrame({"close": [1.0]}, index=pd.date_range("2026-01-05 09:30", periods=1, freq="5min", tz="America/New_York")), CFG)


def test_summarize_win_rate_expectancy_and_drawdown():
    trades = [
        OrbTrade("d1", "long", 1, 2, 2.0, "target"),
        OrbTrade("d2", "long", 1, 0, -1.0, "stop"),
        OrbTrade("d3", "long", 1, 2, 2.0, "target"),
        OrbTrade("d4", "long", 1, 0, -1.0, "stop"),
    ]
    s = summarize(trades)
    assert s["trades"] == 4
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["expectancy_r"] == pytest.approx(0.5)
    assert s["profit_factor"] == pytest.approx(2.0)
    assert s["max_drawdown_r"] == pytest.approx(-1.0)
    assert s["small_sample"] is True


def test_fixed_stop_ignores_the_trail():
    # Bar 8's low ties the trailed stop (101.9), which would stop the trade if the trail were on.
    # With trail=False the stop stays at the range edge (99), so the trade runs to max hold on bar 9.
    cfg = OrbConfig(cost_r=0.0, max_hold_bars=3, trail=False)
    rows = _inside(6)
    rows.append((101.0, 102.5, 100.8, 102.0))        # signal
    rows += [(102.0, 103.0, 101.9, 102.5), (102.5, 103.0, 101.9, 102.5),
             (102.5, 103.0, 101.9, 102.5), (102.5, 103.0, 101.9, 102.6)]
    t = run_orb(_day(rows), cfg)[0]
    assert t.reason == "time"
    assert t.exit == pytest.approx(102.5)            # closes on bar 9


def test_profit_factor_is_none_without_losses_so_json_stays_valid():
    import json
    trades = [OrbTrade("d1", "long", 1, 2, 2.0, "target")]
    s = summarize(trades)
    assert s["profit_factor"] is None
    json.dumps(s, allow_nan=False)                   # raises if any value is inf or nan


def test_summarize_empty_is_safe():
    assert summarize([])["trades"] == 0
