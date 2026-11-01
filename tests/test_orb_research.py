from pathlib import Path

import numpy as np
import pandas as pd

from research.run_orb_research import GATES, VARIANTS, evaluate, split_by_day
from trader.orb import OrbConfig


def _days(n_days):
    """n_days of 20 five-minute bars each, all inside a 99-101 range (so no trades)."""
    frames = []
    for d in range(n_days):
        idx = pd.date_range(f"2026-01-{5 + d:02d} 09:30", periods=20, freq="5min", tz="America/New_York")
        frames.append(pd.DataFrame(np.tile([100.0, 101.0, 99.0, 100.0], (20, 1)), index=idx,
                                   columns=["open", "high", "low", "close"]))
    return pd.concat(frames).tz_convert("UTC")


def test_split_is_by_whole_days_and_in_time_order():
    cfg = OrbConfig()
    df = _days(10)
    train, test = split_by_day(df, cfg, train_fraction=0.7)
    train_days = set(train.index.tz_convert(cfg.tz).date)
    test_days = set(test.index.tz_convert(cfg.tz).date)
    assert train_days and test_days
    assert max(train_days) < min(test_days)          # no day is split across the two sets
    assert len(train) + len(test) == len(df)


def test_no_trades_means_gates_fail():
    df = _days(10)
    result = evaluate("TEST", df, "trail", VARIANTS["trail"])
    assert result["out_of_sample"]["trades"] == 0
    assert result["passed_gates"] is False


def test_gates_are_the_pre_set_values():
    assert GATES == {"min_oos_trades": 30, "min_oos_expectancy_r": 0.0, "max_oos_drawdown_r": -10.0}


def test_research_module_cannot_place_orders():
    source = (Path(__file__).resolve().parent.parent / "research" / "run_orb_research.py").read_text()
    assert "TradingClient" not in source
    assert "submit_order" not in source
    assert "submit_bracket" not in source
