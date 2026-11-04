"""Fixed gold variants I, J, and K.

Variant I is a 5-minute Bollinger/RSI mean-reversion simulator. Variants J and
K are daily gold systems. This module is simulator-only and never imports a
broker or places orders. No parameter search or variant combination is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from research.london_sdt import clean_bars, normalize_bars


@dataclass(frozen=True)
class ReversionConfig:
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    atr_period: int = 14
    sma_exit_period: int = 20
    stop_atr: float = 2.0
    max_bars: int = 48
    cost_r: float = 0.05
    bar_minutes: int = 5

    def __post_init__(self) -> None:
        if self.bb_period < 2:
            raise ValueError("bb_period must be at least 2")
        if self.bb_std <= 0:
            raise ValueError("bb_std must be positive")
        if self.rsi_period < 1:
            raise ValueError("rsi_period must be positive")
        if self.atr_period < 1:
            raise ValueError("atr_period must be positive")
        if self.sma_exit_period < 1:
            raise ValueError("sma_exit_period must be positive")
        if self.stop_atr <= 0:
            raise ValueError("stop_atr must be positive")
        if self.max_bars < 1:
            raise ValueError("max_bars must be positive")
        if self.bar_minutes < 1:
            raise ValueError("bar_minutes must be positive")


@dataclass(frozen=True)
class ReversionTrade:
    day: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    initial_stop: float
    exit: float
    reason: str
    risk: float
    gross_r: float
    cost_r: float
    net_r: float
    bars_held: int


@dataclass
class ReversionResult:
    config: ReversionConfig
    trades: list[ReversionTrade] = field(default_factory=list)
    bars: int = 0
    first: str | None = None
    last: str | None = None
    data_source: str = "unknown"

    def metrics(self) -> dict[str, Any]:
        return summarize_reversion(self.trades)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "Variant I Bollinger/RSI mean reversion",
            "data_source": self.data_source,
            "settings": asdict(self.config),
            "metrics": self.metrics(),
            "bars": self.bars,
            "first": self.first,
            "last": self.last,
            "trades": [asdict(trade) for trade in self.trades],
        }


def add_indicators(bars: pd.DataFrame, config: ReversionConfig | None = None) -> pd.DataFrame:
    config = config or ReversionConfig()
    frame = bars.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    sma = close.rolling(config.bb_period, min_periods=config.bb_period).mean()
    std = close.rolling(config.bb_period, min_periods=config.bb_period).std(ddof=0)
    frame["sma"] = sma
    frame["bb_upper"] = sma + config.bb_std * std
    frame["bb_lower"] = sma - config.bb_std * std
    frame["rsi"] = rsi_wilder(close, config.rsi_period)
    frame["atr"] = atr_wilder(high, low, close, config.atr_period)
    frame["exit_sma"] = close.rolling(
        config.sma_exit_period, min_periods=config.sma_exit_period
    ).mean()
    return frame


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.Series(close, dtype=float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    valid = avg_gain.notna() & avg_loss.notna()
    both_zero = valid & (avg_gain == 0) & (avg_loss == 0)
    only_gain = valid & (avg_loss == 0) & (avg_gain > 0)
    only_loss = valid & (avg_gain == 0) & (avg_loss > 0)
    normal = valid & (avg_gain > 0) & (avg_loss > 0)
    rsi[both_zero] = 50.0
    rsi[only_gain] = 100.0
    rsi[only_loss] = 0.0
    rsi[normal] = 100.0 - 100.0 / (1.0 + avg_gain[normal] / avg_loss[normal])
    return rsi


def atr_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)
    close = pd.Series(close, dtype=float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)
    close = pd.Series(close, dtype=float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_smoothed = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_smoothed = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    atr = atr.replace(0.0, np.nan)
    plus_di = 100.0 * plus_smoothed / atr
    minus_di = 100.0 * minus_smoothed / atr
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def run_reversion(
    bars: pd.DataFrame,
    config: ReversionConfig | None = None,
    data_source: str = "unknown",
    clean: bool = True,
) -> ReversionResult:
    config = config or ReversionConfig()
    frame = clean_bars(bars) if clean else normalize_bars(bars)
    frame = add_indicators(frame, config)
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    lower = frame["bb_lower"].to_numpy(float)
    upper = frame["bb_upper"].to_numpy(float)
    rsi = frame["rsi"].to_numpy(float)
    atr = frame["atr"].to_numpy(float)
    exit_sma = frame["exit_sma"].to_numpy(float)
    index = frame.index
    n = len(frame)
    trades: list[ReversionTrade] = []
    i = 0
    while i < n - 1:
        long_signal = closes[i] < lower[i] and rsi[i] < 30.0
        short_signal = closes[i] > upper[i] and rsi[i] > 70.0
        if not (long_signal or short_signal) or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        side = "long" if long_signal else "short"
        entry_bar = i + 1
        entry = float(opens[entry_bar])
        risk = config.stop_atr * float(atr[i])
        if entry <= 0 or risk <= 0:
            i += 1
            continue
        stop = entry - risk if side == "long" else entry + risk
        exit_price, exit_bar, reason = _manage_reversion(
            side,
            entry_bar,
            entry,
            stop,
            exit_sma,
            opens,
            highs,
            lows,
            closes,
            config,
        )
        sign = 1.0 if side == "long" else -1.0
        gross_r = sign * (exit_price - entry) / risk
        net_r = gross_r - config.cost_r
        trades.append(
            ReversionTrade(
                day=str(index[entry_bar].date()),
                side=side,
                entry_time=str(index[entry_bar]),
                exit_time=str(index[exit_bar] + pd.Timedelta(minutes=config.bar_minutes)),
                entry=entry,
                initial_stop=float(stop),
                exit=float(exit_price),
                reason=reason,
                risk=float(risk),
                gross_r=float(gross_r),
                cost_r=config.cost_r,
                net_r=float(net_r),
                bars_held=exit_bar - entry_bar + 1,
            )
        )
        i = exit_bar

    return ReversionResult(
        config=config,
        trades=trades,
        bars=n,
        first=str(index[0]) if n else None,
        last=str(index[-1]) if n else None,
        data_source=data_source,
    )


def _manage_reversion(
    side: str,
    entry_bar: int,
    entry: float,
    stop: float,
    exit_sma: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    config: ReversionConfig,
) -> tuple[float, int, str]:
    long = side == "long"
    n = len(closes)
    for j in range(entry_bar, n):
        sma = exit_sma[j]
        if long:
            if opens[j] <= stop:
                return float(opens[j]), j, "stop_gap"
            if np.isfinite(sma) and lows[j] <= stop and highs[j] >= sma:
                return float(stop), j, "stop"
            if lows[j] <= stop:
                return float(stop), j, "stop"
            if np.isfinite(sma):
                if opens[j] >= sma:
                    return float(opens[j]), j, "sma_gap"
                if highs[j] >= sma:
                    return float(sma), j, "sma"
        else:
            if opens[j] >= stop:
                return float(opens[j]), j, "stop_gap"
            if np.isfinite(sma) and highs[j] >= stop and lows[j] <= sma:
                return float(stop), j, "stop"
            if highs[j] >= stop:
                return float(stop), j, "stop"
            if np.isfinite(sma):
                if opens[j] <= sma:
                    return float(opens[j]), j, "sma_gap"
                if lows[j] <= sma:
                    return float(sma), j, "sma"
        if j - entry_bar + 1 >= config.max_bars:
            return float(closes[j]), j, "time"
        if j == n - 1:
            return float(closes[j]), j, "data_end"
    return float(closes[-1]), n - 1, "data_end"


def summarize_reversion(trades: list[ReversionTrade]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "worst_loss_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "max_drawdown_r": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "stop_exits": 0,
            "sma_exits": 0,
            "time_exits": 0,
            "data_end_exits": 0,
            "small_sample": True,
        }
    rs = np.array([trade.net_r for trade in trades], float)
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    cumulative = np.cumsum(rs)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    reasons = [trade.reason for trade in trades]
    return {
        "trades": len(rs),
        "win_rate": float((rs > 0).mean()),
        "avg_win_r": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_r": float(losses.mean()) if losses.size else 0.0,
        "worst_loss_r": float(losses.min()) if losses.size else 0.0,
        "expectancy_r": float(rs.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size else None,
        "total_r": float(rs.sum()),
        "max_drawdown_r": float((cumulative - peak).min()),
        "long_trades": sum(trade.side == "long" for trade in trades),
        "short_trades": sum(trade.side == "short" for trade in trades),
        "stop_exits": sum(reason in {"stop", "stop_gap"} for reason in reasons),
        "sma_exits": sum(reason in {"sma", "sma_gap"} for reason in reasons),
        "time_exits": sum(reason == "time" for reason in reasons),
        "data_end_exits": sum(reason == "data_end" for reason in reasons),
        "small_sample": len(rs) < 30,
    }


def reversion_with_cost(result: ReversionResult, cost_r: float) -> list[ReversionTrade]:
    return [
        replace(trade, cost_r=cost_r, net_r=trade.gross_r - cost_r)
        for trade in result.trades
    ]


@dataclass(frozen=True)
class DailyJConfig:
    sma_period: int = 110
    adx_period: int = 14
    adx_min: float = 20.0


@dataclass(frozen=True)
class DailyTrade:
    entry_date: str
    exit_date: str
    side: str
    entry: float
    exit: float
    return_pct: float
    bars_held: int
    reason: str


@dataclass
class DailyJResult:
    config: DailyJConfig
    trades: list[DailyTrade] = field(default_factory=list)
    bars: int = 0
    first: str | None = None
    last: str | None = None
    data_source: str = "unknown"

    def metrics(self) -> dict[str, Any]:
        return summarize_daily_trades(self.trades)

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "Variant J daily SMA/ADX trend",
            "data_source": self.data_source,
            "settings": asdict(self.config),
            "metrics": self.metrics(),
            "bars": self.bars,
            "first": self.first,
            "last": self.last,
            "trades": [asdict(trade) for trade in self.trades],
        }


def add_daily_j_indicators(
    bars: pd.DataFrame,
    config: DailyJConfig | None = None,
) -> pd.DataFrame:
    config = config or DailyJConfig()
    frame = bars.copy()
    close = frame["close"].astype(float)
    frame["sma"] = close.rolling(config.sma_period, min_periods=config.sma_period).mean()
    frame["adx"] = adx_wilder(
        frame["high"],
        frame["low"],
        close,
        config.adx_period,
    )
    return frame


def run_daily_j(
    bars: pd.DataFrame,
    config: DailyJConfig | None = None,
    data_source: str = "unknown",
) -> DailyJResult:
    config = config or DailyJConfig()
    frame = normalize_bars(bars)
    frame = add_daily_j_indicators(frame, config)
    opens = frame["open"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    sma = frame["sma"].to_numpy(float)
    adx = frame["adx"].to_numpy(float)
    index = frame.index
    n = len(frame)
    bullish = closes > opens
    bearish = closes < opens
    long_ok = bullish & (closes > sma) & (adx >= config.adx_min)
    short_ok = bearish & (closes < sma) & (adx >= config.adx_min)
    trades: list[DailyTrade] = []
    position: str | None = None
    entry_bar: int | None = None
    entry_price: float | None = None

    for bar in range(n - 1):
        if position is None:
            if long_ok[bar]:
                position = "long"
                entry_bar = bar + 1
                entry_price = float(opens[entry_bar])
            elif short_ok[bar]:
                position = "short"
                entry_bar = bar + 1
                entry_price = float(opens[entry_bar])
            continue

        still_ok = long_ok[bar] if position == "long" else short_ok[bar]
        if still_ok:
            continue

        exit_bar = bar + 1
        trades.append(
            _daily_trade(
                index,
                position,
                entry_bar,
                entry_price,
                float(opens[exit_bar]),
                exit_bar,
                "condition",
            )
        )
        position = None
        entry_bar = None
        entry_price = None
        if long_ok[bar]:
            position = "long"
            entry_bar = exit_bar
            entry_price = float(opens[exit_bar])
        elif short_ok[bar]:
            position = "short"
            entry_bar = exit_bar
            entry_price = float(opens[exit_bar])

    if position is not None and entry_bar is not None and entry_price is not None:
        trades.append(
            _daily_trade(
                index,
                position,
                entry_bar,
                entry_price,
                float(closes[-1]),
                n - 1,
                "data_end",
            )
        )

    return DailyJResult(
        config=config,
        trades=trades,
        bars=n,
        first=str(index[0]) if n else None,
        last=str(index[-1]) if n else None,
        data_source=data_source,
    )


def _daily_trade(
    index: pd.DatetimeIndex,
    side: str,
    entry_bar: int,
    entry: float | None,
    exit_price: float,
    exit_bar: int,
    reason: str,
) -> DailyTrade:
    if entry is None or entry <= 0:
        raise ValueError("daily trade entry must be positive")
    sign = 1.0 if side == "long" else -1.0
    return DailyTrade(
        entry_date=str(index[entry_bar].date()),
        exit_date=str(index[exit_bar].date()),
        side=side,
        entry=float(entry),
        exit=float(exit_price),
        return_pct=float(100.0 * sign * (exit_price / entry - 1.0)),
        bars_held=exit_bar - entry_bar + 1,
        reason=reason,
    )


def summarize_daily_trades(trades: list[DailyTrade]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "worst_loss_pct": 0.0,
            "expectancy_pct": 0.0,
            "profit_factor": 0.0,
            "total_return_pct": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "data_end_exits": 0,
            "small_sample": True,
        }
    returns = np.array([trade.return_pct for trade in trades], float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    return {
        "trades": len(returns),
        "win_rate": float((returns > 0).mean()),
        "avg_win_pct": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_pct": float(losses.mean()) if losses.size else 0.0,
        "worst_loss_pct": float(losses.min()) if losses.size else 0.0,
        "expectancy_pct": float(returns.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size else None,
        "total_return_pct": float((np.prod(1.0 + returns / 100.0) - 1.0) * 100.0),
        "long_trades": sum(trade.side == "long" for trade in trades),
        "short_trades": sum(trade.side == "short" for trade in trades),
        "data_end_exits": sum(trade.reason == "data_end" for trade in trades),
        "small_sample": len(returns) < 30,
    }


@dataclass(frozen=True)
class DailyKConfig:
    sma_period: int = 250


@dataclass
class DailyKResult:
    config: DailyKConfig
    yearly: dict[str, dict[str, Any]]
    metrics: dict[str, Any]
    entries: list[str]
    bars: int
    first: str | None
    last: str | None
    data_source: str = "unknown"

    def report(self) -> dict[str, Any]:
        return {
            "strategy": "Variant K daily SMA trend long/cash",
            "data_source": self.data_source,
            "settings": asdict(self.config),
            "metrics": self.metrics,
            "yearly": self.yearly,
            "entries": self.entries,
            "bars": self.bars,
            "first": self.first,
            "last": self.last,
        }


def run_daily_k(
    bars: pd.DataFrame,
    config: DailyKConfig | None = None,
    data_source: str = "unknown",
) -> DailyKResult:
    config = config or DailyKConfig()
    frame = normalize_bars(bars)
    close = frame["close"].astype(float)
    sma = close.rolling(config.sma_period, min_periods=config.sma_period).mean()
    invested = close > sma
    valid_sma = sma.notna().to_numpy()
    if not valid_sma.any():
        raise ValueError("not enough daily bars for the SMA period")
    first_sma_pos = int(np.flatnonzero(valid_sma)[0])
    start_pos = first_sma_pos + 1
    if start_pos >= len(frame):
        raise ValueError("not enough daily bars after the first SMA value")
    period = frame.iloc[start_pos:]
    period_returns = close.pct_change().iloc[start_pos:]
    position = invested.shift(1).iloc[start_pos:].fillna(False).astype(float)
    strategy_returns = position * period_returns
    entries = _entry_dates(period.index, position)
    yearly = _daily_k_yearly(period.index, strategy_returns, period_returns, position, entries)
    equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    metrics = {
        "start": str(period.index[0]) if len(period) else None,
        "end": str(period.index[-1]) if len(period) else None,
        "bars": int(len(period)),
        "entries": len(entries),
        "days_invested": int(position.sum()),
        "strategy_return_pct": float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) else 0.0,
        "buy_hold_return_pct": float(
            (np.prod(1.0 + period_returns.fillna(0.0)) - 1.0) * 100.0
        )
        if len(period_returns)
        else 0.0,
        "strategy_max_drawdown_pct": float(drawdown.min() * 100.0) if len(drawdown) else 0.0,
    }
    return DailyKResult(
        config=config,
        yearly=yearly,
        metrics=metrics,
        entries=entries,
        bars=len(frame),
        first=str(frame.index[0]) if len(frame) else None,
        last=str(frame.index[-1]) if len(frame) else None,
        data_source=data_source,
    )


def _entry_dates(index: pd.DatetimeIndex, position: pd.Series) -> list[str]:
    previous = position.shift(1).fillna(0.0)
    entered = (position == 1.0) & (previous == 0.0)
    return [str(index[position].date()) for position in np.flatnonzero(entered.to_numpy())]


def _daily_k_yearly(
    index: pd.DatetimeIndex,
    strategy_returns: pd.Series,
    buy_hold_returns: pd.Series,
    position: pd.Series,
    entries: list[str],
) -> dict[str, dict[str, Any]]:
    years = sorted({timestamp.year for timestamp in index})
    out: dict[str, dict[str, Any]] = {}
    for year in years:
        mask = index.year == year
        strategy_year = strategy_returns[mask].fillna(0.0)
        buy_hold_year = buy_hold_returns[mask].fillna(0.0)
        year_entries = sum(entry.startswith(str(year)) for entry in entries)
        out[str(year)] = {
            "entries": year_entries,
            "days_invested": int(position[mask].sum()),
            "strategy_return_pct": float(
                (np.prod(1.0 + strategy_year) - 1.0) * 100.0
            )
            if len(strategy_year)
            else 0.0,
            "buy_hold_return_pct": float(
                (np.prod(1.0 + buy_hold_year) - 1.0) * 100.0
            )
            if len(buy_hold_year)
            else 0.0,
        }
    return out
