"""Event-driven, long-only backtest for the intraday noise-band strategy.

Conservative assumptions:
- A signal at a bar close fills at the next bar's open.
- Slippage is charged on both sides; commission is per share per side.
- A stop checked at a half-hourly decision point fills at the next open, so a
  gap through the stop is not hidden.
- The mandatory flat exit fills at the signal bar's close.
- Position size is recomputed at the fill price, not the signal price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bands import (
    Bands,
    bands_for_slot,
    group_sessions,
    is_decision_slot,
    prepare_session,
    session_close,
    session_open,
    sigma_by_slot,
)
from .config import IntradaySettings
from .strategy import size_position


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    sessions: int


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_id", "date", "entry_ts", "exit_ts", "side", "qty",
            "entry_price", "exit_price", "stop", "pnl", "ret", "r_multiple",
            "reason", "hold_minutes", "mae", "mfe",
        ]
    )


def run_backtest(
    bars: pd.DataFrame,
    s: IntradaySettings | None = None,
    starting_equity: float = 100_000.0,
) -> BacktestResult:
    s = s or IntradaySettings()
    sessions = group_sessions(bars)
    ordered = sorted(sessions.items())
    if len(ordered) <= s.band_lookback_sessions:
        return BacktestResult(_empty_trades(), pd.Series(dtype=float), _metrics(pd.Series(dtype=float), _empty_trades(), starting_equity, len(ordered)), len(ordered))

    slip = s.slippage_bps / 10_000.0
    commission = s.commission_per_share
    flat_slot = s.flat_time.hour * 60 + s.flat_time.minute
    equity = starting_equity
    curve: list[tuple[str, float]] = []
    trades: list[dict] = []

    for index, (day, raw) in enumerate(ordered):
        if index < s.band_lookback_sessions:
            curve.append((str(day), equity))
            continue

        prior = [frame for _, frame in ordered[max(0, index - s.band_lookback_sessions):index]]
        sigma = sigma_by_slot(prior, s.bar_minutes, s.band_lookback_sessions)
        prior_close = session_close(prior[-1]) if prior else float(raw["open"].iloc[0])
        open_price = session_open(raw)
        frame = prepare_session(raw, s.bar_minutes)
        if frame.empty:
            curve.append((str(day), equity))
            continue

        opens = frame["open"].to_numpy(float)
        highs = frame["high"].to_numpy(float)
        lows = frame["low"].to_numpy(float)
        closes = frame["close"].to_numpy(float)
        slots = frame["slot"].to_numpy(int)
        vwaps = frame["vwap"].to_numpy(float)
        stamps = list(frame.index)

        position: dict | None = None
        pending: dict | None = None
        trades_today = 0
        min_price = max_price = 0.0

        for i in range(len(frame)):
            # 1) Fill anything queued by the previous bar at this bar's open.
            if pending is not None:
                action = pending["action"]
                if action == "exit" and position is not None:
                    exit_price = opens[i] * (1.0 - slip)
                    _close(
                        trades, position, exit_price, stamps[i], pending["reason"],
                        commission, min_price, max_price, day,
                    )
                    equity += trades[-1]["pnl"]
                    position = None
                elif action == "enter" and position is None:
                    fill = opens[i] * (1.0 + slip)
                    stop = pending["stop"]
                    if stop < fill:
                        sized = size_position(equity, fill, stop, s)
                        if sized is not None:
                            qty, _ = sized
                            position = {
                                "trade_id": f"bt-{day}-{trades_today + 1}",
                                "qty": qty,
                                "entry_price": fill,
                                "entry_ts": stamps[i],
                                "stop": stop,
                                "initial_stop": stop,
                            }
                            min_price = max_price = fill
                            trades_today += 1
                pending = None

            bands = bands_for_slot(open_price, prior_close, sigma, int(slots[i]), s.band_mult)
            if bands is None:
                continue
            price = closes[i]
            vwap = vwaps[i]

            if position is not None:
                min_price = min(min_price, lows[i])
                max_price = max(max_price, highs[i])
                slot = int(slots[i])
                if slot >= flat_slot:
                    exit_price = price * (1.0 - slip)
                    _close(
                        trades, position, exit_price, stamps[i], "flat_by_close",
                        commission, min_price, max_price, day,
                    )
                    equity += trades[-1]["pnl"]
                    position = None
                elif is_decision_slot(slot, s.entry_start, s.entry_end, s.decision_minutes):
                    stop = max(position["stop"], bands.upper, vwap)
                    position["stop"] = stop
                    if price < stop:
                        pending = {"action": "exit", "reason": "trail_stop"}
                continue

            if pending is None and is_decision_slot(int(slots[i]), s.entry_start, s.entry_end, s.decision_minutes):
                if trades_today < s.max_trades_per_day and price > bands.upper:
                    stop = max(bands.upper, vwap)
                    if stop < price:
                        pending = {"action": "enter", "stop": stop}

        if position is not None:
            exit_price = closes[-1] * (1.0 - slip)
            _close(
                trades, position, exit_price, stamps[-1], "session_end",
                commission, min_price, max_price, day,
            )
            equity += trades[-1]["pnl"]
            position = None

        curve.append((str(day), equity))

    trade_frame = pd.DataFrame(trades) if trades else _empty_trades()
    equity_series = pd.Series(
        [value for _, value in curve],
        index=pd.Index([day for day, _ in curve], name="date"),
        name="equity",
    )
    return BacktestResult(trade_frame, equity_series, _metrics(equity_series, trade_frame, starting_equity, len(ordered)), len(ordered))


def _close(
    trades: list[dict],
    position: dict,
    exit_price: float,
    exit_ts,
    reason: str,
    commission: float,
    min_price: float,
    max_price: float,
    day,
) -> None:
    qty = int(position["qty"])
    entry = float(position["entry_price"])
    pnl = (exit_price - entry) * qty - commission * 2 * qty
    risk_per_share = entry - float(position["initial_stop"])
    r_multiple = (exit_price - entry) / risk_per_share if risk_per_share > 0 else 0.0
    entry_ts = pd.Timestamp(position["entry_ts"])
    exit_ts = pd.Timestamp(exit_ts)
    hold = (exit_ts - entry_ts).total_seconds() / 60.0
    trades.append(
        {
            "trade_id": position["trade_id"],
            "date": str(day),
            "entry_ts": entry_ts.isoformat(),
            "exit_ts": exit_ts.isoformat(),
            "side": "long",
            "qty": qty,
            "entry_price": entry,
            "exit_price": float(exit_price),
            "stop": float(position["initial_stop"]),
            "pnl": float(pnl),
            "ret": float(exit_price / entry - 1.0),
            "r_multiple": float(r_multiple),
            "reason": reason,
            "hold_minutes": float(hold),
            "mae": float((min_price - entry) / risk_per_share) if risk_per_share > 0 else 0.0,
            "mfe": float((max_price - entry) / risk_per_share) if risk_per_share > 0 else 0.0,
        }
    )


def _metrics(equity: pd.Series, trades: pd.DataFrame, starting_equity: float, sessions: int) -> dict:
    if equity.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "profit_factor": None, "expectancy": 0.0,
            "avg_r": 0.0, "avg_hold_minutes": 0.0, "sessions": sessions,
            "gates": {}, "yearly": [], "small_sample": True,
        }
    returns = equity.pct_change().dropna()
    std = float(returns.std()) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    total_return = float(equity.iloc[-1] / starting_equity - 1.0)
    years = max(len(equity) / 252.0, 1e-9)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    pnls = trades["pnl"].astype(float) if not trades.empty else pd.Series(dtype=float)
    rs = trades["r_multiple"].astype(float) if not trades.empty else pd.Series(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    yearly = []
    if not trades.empty:
        frame = trades.copy()
        frame["year"] = frame["date"].str[:4]
        for year, group in frame.groupby("year", sort=True):
            group_pnl = group["pnl"].astype(float)
            yearly.append(
                {
                    "year": str(year),
                    "pnl": float(group_pnl.sum()),
                    "trades": int(len(group)),
                    "win_rate": float((group_pnl > 0).mean()),
                }
            )
    gates = {
        "trades>=100": len(trades) >= 100,
        "sharpe>=0.5": sharpe >= 0.5,
        "return>0": total_return > 0,
        "max_dd>-20%": float(drawdown.min()) > -0.20,
        "profit_factor>=1.2": (gross_profit / gross_loss >= 1.2) if gross_loss > 0 else False,
        "avg_r>0": float(rs.mean()) > 0 if len(rs) else False,
    }
    return {
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else 0.0,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy": float(pnls.mean()) if len(pnls) else 0.0,
        "avg_r": float(rs.mean()) if len(rs) else 0.0,
        "avg_hold_minutes": float(trades["hold_minutes"].mean()) if not trades.empty else 0.0,
        "sessions": sessions,
        "gates": gates,
        "yearly": yearly,
        "small_sample": len(trades) < 100,
    }
