"""Heikin Ashi swing strategy simulator.

Rules are frozen in ``research/ha_swing_rules.md``. This module is read-only
research code and never imports a broker or places orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

OHLCV = ("open", "high", "low", "close", "volume")
BARS_PER_HOUR = 12
AVG_BODY_PERIOD = 10
COST_BPS = (10.0, 20.0)
TRAIN_FRACTION = 0.70


@dataclass(frozen=True)
class HaTrade:
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    risk: float
    gross_r: float
    cost_r_10bp: float
    cost_r_20bp: float
    net_r_10bp: float
    net_r_20bp: float
    bars_held: int
    exit_reason: str


def build_hourly_bars(
    five_min: pd.DataFrame,
    bars_per_hour: int = BARS_PER_HOUR,
) -> pd.DataFrame:
    """Build complete UTC one-hour bars from five-minute bars.

    Synthetic flat padding bars are removed before aggregation because they
    represent closed-market periods rather than real prices.
    """
    if five_min is None or five_min.empty:
        raise ValueError("five-minute data is empty")

    missing = [column for column in OHLCV if column not in five_min.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    input_rows = len(five_min)
    frame = five_min.loc[:, list(OHLCV)].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if frame.empty:
        raise ValueError("no usable five-minute bars")

    flat = (
        (frame["open"] == frame["high"])
        & (frame["high"] == frame["low"])
        & (frame["low"] == frame["close"])
    )
    flat_count = int(flat.sum())
    real = frame.loc[~flat]
    if real.empty:
        raise ValueError("no real five-minute bars after removing flat padding")

    grouped = real.resample("1h", label="left", closed="left")
    hourly = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped.size()
    complete = counts == bars_per_hour
    incomplete = int(((counts > 0) & ~complete).sum())
    hourly = hourly.loc[complete].dropna(subset=["open", "high", "low", "close"])
    if hourly.empty:
        raise ValueError("no complete one-hour bars")

    hourly.index = pd.DatetimeIndex(hourly.index).rename(None)
    hourly.attrs.update(
        five_min_rows=input_rows,
        real_five_min_rows=len(real),
        flat_padding_removed=flat_count,
        complete_hourly_bars=len(hourly),
        incomplete_hours_dropped=incomplete,
        bars_per_hour=bars_per_hour,
    )
    return hourly


def heikin_ashi(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add standard Heikin Ashi columns to complete one-hour bars."""
    if hourly is None or hourly.empty:
        raise ValueError("hourly bars are empty")

    frame = hourly.copy()
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)

    ha_close = (open_ + high + low + close) / 4.0
    ha_open = np.empty(len(frame), dtype=float)
    ha_open[0] = (open_[0] + close[0]) / 2.0
    for index in range(1, len(frame)):
        ha_open[index] = (ha_open[index - 1] + ha_close[index - 1]) / 2.0

    frame["ha_open"] = ha_open
    frame["ha_close"] = ha_close
    frame["ha_high"] = np.maximum(high, np.maximum(ha_open, ha_close))
    frame["ha_low"] = np.minimum(low, np.minimum(ha_open, ha_close))
    return frame


def add_signals(
    ha: pd.DataFrame,
    avg_period: int = AVG_BODY_PERIOD,
) -> pd.DataFrame:
    """Add body statistics and frozen long/short signal columns."""
    frame = ha.copy()
    body = frame["ha_close"] - frame["ha_open"]
    abs_body = body.abs()
    avg_body = abs_body.rolling(avg_period, min_periods=avg_period).mean()

    frame["body"] = body
    frame["abs_body"] = abs_body
    frame["avg_body_10"] = avg_body
    frame["long_signal"] = (
        (body > 0)
        & (frame["ha_open"] == frame["ha_low"])
        & (abs_body > avg_body)
    )
    frame["short_signal"] = (
        (body < 0)
        & (frame["ha_open"] == frame["ha_high"])
        & (abs_body > avg_body)
    )
    return frame


def split_days(
    index: pd.DatetimeIndex,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split unique UTC calendar days into whole-day train and test sets."""
    days = pd.DatetimeIndex(index.normalize().unique()).sort_values()
    if len(days) < 2:
        raise ValueError("need at least two unique days")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")

    train_count = int(len(days) * train_fraction)
    train_count = max(1, min(train_count, len(days) - 1))
    return days[:train_count], days[train_count:]


def _cost_r(entry_price: float, risk: float, cost_bp: float) -> float:
    return (entry_price * cost_bp / 10000.0) / risk


def _make_trade(
    side: str,
    signal_index: int,
    entry_index: int,
    exit_index: int,
    entry_price: float,
    exit_price: float,
    risk: float,
    exit_reason: str,
    index: pd.DatetimeIndex,
) -> HaTrade:
    if side == "long":
        gross_r = (exit_price - entry_price) / risk
    elif side == "short":
        gross_r = (entry_price - exit_price) / risk
    else:
        raise ValueError(f"unsupported side: {side}")

    cost_10 = _cost_r(entry_price, risk, COST_BPS[0])
    cost_20 = _cost_r(entry_price, risk, COST_BPS[1])
    return HaTrade(
        side=side,
        signal_time=str(index[signal_index]),
        entry_time=str(index[entry_index]),
        exit_time=str(index[exit_index]),
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        risk=float(risk),
        gross_r=float(gross_r),
        cost_r_10bp=float(cost_10),
        cost_r_20bp=float(cost_20),
        net_r_10bp=float(gross_r - cost_10),
        net_r_20bp=float(gross_r - cost_20),
        bars_held=int(exit_index - entry_index),
        exit_reason=exit_reason,
    )


def simulate(ha: pd.DataFrame) -> list[HaTrade]:
    """Run the frozen one-position-at-a-time swing simulator."""
    required = {
        "open",
        "high",
        "low",
        "close",
        "ha_open",
        "ha_high",
        "ha_low",
        "ha_close",
        "long_signal",
        "short_signal",
    }
    missing = required - set(ha.columns)
    if missing:
        raise ValueError(f"missing signal columns: {', '.join(sorted(missing))}")
    if ha.empty:
        return []

    frame = ha
    index = pd.DatetimeIndex(frame.index)
    trades: list[HaTrade] = []
    position: dict[str, Any] | None = None
    last_index = len(frame) - 1

    for i in range(len(frame)):
        if position is None:
            if i == last_index:
                break
            if bool(frame["long_signal"].iloc[i]):
                side = "long"
            elif bool(frame["short_signal"].iloc[i]):
                side = "short"
            else:
                continue

            risk = float(frame["ha_high"].iloc[i] - frame["ha_low"].iloc[i])
            if not np.isfinite(risk) or risk <= 0:
                continue

            entry_index = i + 1
            entry_price = float(frame["open"].iloc[entry_index])
            if not np.isfinite(entry_price):
                continue
            position = {
                "side": side,
                "signal_index": i,
                "entry_index": entry_index,
                "entry_price": entry_price,
                "risk": risk,
            }
            continue

        side = str(position["side"])
        ha_open = float(frame["ha_open"].iloc[i])
        ha_close = float(frame["ha_close"].iloc[i])
        opposite = (side == "long" and ha_close < ha_open) or (
            side == "short" and ha_close > ha_open
        )
        if not opposite:
            continue

        if i + 1 < len(frame):
            exit_index = i + 1
            exit_price = float(frame["open"].iloc[exit_index])
            exit_reason = "opposite_color"
        else:
            exit_index = i
            exit_price = float(frame["close"].iloc[i])
            exit_reason = "opposite_color_end_of_data"

        trades.append(
            _make_trade(
                side=side,
                signal_index=int(position["signal_index"]),
                entry_index=int(position["entry_index"]),
                exit_index=exit_index,
                entry_price=float(position["entry_price"]),
                exit_price=exit_price,
                risk=float(position["risk"]),
                exit_reason=exit_reason,
                index=index,
            )
        )
        position = None

    if position is not None:
        exit_index = last_index
        trades.append(
            _make_trade(
                side=str(position["side"]),
                signal_index=int(position["signal_index"]),
                entry_index=int(position["entry_index"]),
                exit_index=exit_index,
                entry_price=float(position["entry_price"]),
                exit_price=float(frame["close"].iloc[exit_index]),
                risk=float(position["risk"]),
                exit_reason="end_of_data",
                index=index,
            )
        )

    return trades


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.concatenate(([0.0], np.cumsum(values)))
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def summarize_trades(
    trades: list[HaTrade],
    side: str | None = None,
) -> dict[str, Any]:
    selected = [trade for trade in trades if side is None or trade.side == side]
    if not selected:
        return {
            "trades": 0,
            "win_rate_10bp": 0.0,
            "win_rate_gross": 0.0,
            "avg_gross_r": 0.0,
            "expectancy_10bp": 0.0,
            "expectancy_20bp": 0.0,
            "worst_trade_10bp": None,
            "worst_trade_gross": None,
            "max_drawdown_10bp": 0.0,
            "max_drawdown_20bp": 0.0,
            "total_r_10bp": 0.0,
            "total_r_20bp": 0.0,
            "avg_win_10bp": 0.0,
            "avg_loss_10bp": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }

    gross = np.array([trade.gross_r for trade in selected], dtype=float)
    net_10 = np.array([trade.net_r_10bp for trade in selected], dtype=float)
    net_20 = np.array([trade.net_r_20bp for trade in selected], dtype=float)
    wins = net_10[net_10 > 0]
    losses = net_10[net_10 < 0]

    return {
        "trades": len(selected),
        "win_rate_10bp": float((net_10 > 0).mean() * 100.0),
        "win_rate_gross": float((gross > 0).mean() * 100.0),
        "avg_gross_r": float(gross.mean()),
        "expectancy_10bp": float(net_10.mean()),
        "expectancy_20bp": float(net_20.mean()),
        "worst_trade_10bp": float(net_10.min()),
        "worst_trade_gross": float(gross.min()),
        "max_drawdown_10bp": _max_drawdown(net_10),
        "max_drawdown_20bp": _max_drawdown(net_20),
        "total_r_10bp": float(net_10.sum()),
        "total_r_20bp": float(net_20.sum()),
        "avg_win_10bp": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_10bp": float(losses.mean()) if losses.size else 0.0,
        "long_trades": sum(trade.side == "long" for trade in selected),
        "short_trades": sum(trade.side == "short" for trade in selected),
    }


def summary_block(trades: list[HaTrade]) -> dict[str, Any]:
    return {
        "all": summarize_trades(trades),
        "long": summarize_trades(trades, "long"),
        "short": summarize_trades(trades, "short"),
    }


def buy_and_hold_return(
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    window = bars.loc[(bars.index >= start) & (bars.index <= end)]
    if len(window) < 2:
        return 0.0
    first_close = float(window["close"].iloc[0])
    last_close = float(window["close"].iloc[-1])
    if first_close == 0.0:
        return 0.0
    return (last_close / first_close - 1.0) * 100.0


def run_research(
    five_min: pd.DataFrame,
    include_out_of_sample: bool = True,
    train_fraction: float = TRAIN_FRACTION,
) -> dict[str, Any]:
    hourly = build_hourly_bars(five_min)
    ha = add_signals(heikin_ashi(hourly))
    train_days, test_days = split_days(ha.index, train_fraction)

    train_mask = ha.index.normalize().isin(train_days)
    train_bars = ha.loc[train_mask]
    train_trades = simulate(train_bars)

    metadata = {
        "strategy": "Heikin Ashi swing",
        "rules_file": "research/ha_swing_rules.md",
        "data_file": "data/XAUUSD_5m.csv",
        "actual_start": str(ha.index[0]),
        "actual_end": str(ha.index[-1]),
        "five_min_rows": int(hourly.attrs.get("five_min_rows", 0)),
        "real_five_min_rows": int(hourly.attrs.get("real_five_min_rows", 0)),
        "flat_padding_removed": int(hourly.attrs.get("flat_padding_removed", 0)),
        "complete_hourly_bars": int(len(hourly)),
        "incomplete_hours_dropped": int(hourly.attrs.get("incomplete_hours_dropped", 0)),
        "unique_days": int(len(pd.DatetimeIndex(ha.index.normalize().unique()))),
        "bars_per_hour": BARS_PER_HOUR,
        "avg_body_period": AVG_BODY_PERIOD,
        "cost_bp": [int(value) for value in COST_BPS],
        "train_fraction": float(train_fraction),
        "in_sample_days": int(len(train_days)),
        "in_sample_start": str(train_bars.index[0]),
        "in_sample_end": str(train_bars.index[-1]),
        "data_notes": [
            "The requested 2020-2026 span is not present in data/XAUUSD_5m.csv; results use the actual file coverage.",
            f"Removed {int(hourly.attrs.get('flat_padding_removed', 0))} synthetic flat five-minute padding bars.",
            f"Dropped {int(hourly.attrs.get('incomplete_hours_dropped', 0))} incomplete one-hour bars.",
        ],
    }

    report: dict[str, Any] = {
        "metadata": metadata,
        "in_sample": {
            "metrics": summary_block(train_trades),
            "trades": [asdict(trade) for trade in train_trades],
        },
    }

    if include_out_of_sample:
        test_mask = ha.index.normalize().isin(test_days)
        test_bars = ha.loc[test_mask]
        test_trades = simulate(test_bars)
        metadata.update(
            {
                "out_of_sample_days": int(len(test_days)),
                "out_of_sample_start": str(test_bars.index[0]),
                "out_of_sample_end": str(test_bars.index[-1]),
            }
        )
        report["out_of_sample"] = {
            "metrics": summary_block(test_trades),
            "trades": [asdict(trade) for trade in test_trades],
            "buy_and_hold_return_pct": buy_and_hold_return(
                test_bars, test_bars.index[0], test_bars.index[-1]
            ),
        }

    return report
