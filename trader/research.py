"""Choose, for each symbol, the strategy that held up best on data it was not tuned on.

Honest caveat: picking the best of 16 combinations, even out-of-sample, still carries
selection bias. The min-trades, positive-return and min-Sharpe gates reduce that risk
but do not remove it. Past backtests do not guarantee future results.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import Metrics, evaluate_oos
from .config import Settings
from .strategies import STRATEGIES

MIN_BARS = 120
COLUMNS = ["symbol", "strategy", "oos_return_%", "oos_sharpe", "max_dd_%", "trades", "win_rate_%"]


@dataclass(frozen=True)
class Candidate:
    symbol: str
    strategy: str
    oos: Metrics


def bars_per_year(s: Settings) -> float:
    return 252 * 6.5 / s.timeframe_hours  # trading days x regular session hours


def evaluate_all(bars: dict[str, pd.DataFrame], s: Settings) -> list[Candidate]:
    results: list[Candidate] = []
    for symbol, df in bars.items():
        if len(df) < MIN_BARS:
            continue
        for name, make_signals in STRATEGIES.items():
            metrics = evaluate_oos(df, make_signals(df), s, bars_per_year(s))
            results.append(Candidate(symbol, name, metrics))
    return results


def qualifying_best(candidates: list[Candidate], s: Settings) -> dict[str, Candidate]:
    """Best strategy per symbol that clears every gate. NaN fails each gate."""
    best: dict[str, Candidate] = {}
    for c in candidates:
        m = c.oos
        if m.trades < s.min_oos_trades or not (m.total_return > 0) or not (m.sharpe >= s.min_oos_sharpe):
            continue
        current = best.get(c.symbol)
        if current is None or m.sharpe > current.oos.sharpe:
            best[c.symbol] = c
    return best


def report_frame(candidates: list[Candidate]) -> pd.DataFrame:
    rows = [
        {
            "symbol": c.symbol,
            "strategy": c.strategy,
            "oos_return_%": round(c.oos.total_return * 100, 2),
            "oos_sharpe": round(c.oos.sharpe, 2),
            "max_dd_%": round(c.oos.max_drawdown * 100, 2),
            "trades": c.oos.trades,
            "win_rate_%": round(c.oos.win_rate * 100, 1),
        }
        for c in candidates
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.sort_values(["symbol", "oos_sharpe"], ascending=[True, False]).reset_index(drop=True)
