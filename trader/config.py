"""Settings, read from environment variables with conservative defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_UNIVERSE = ("SPY", "GLD", "AAPL", "TSLA")


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    api_secret: str = ""
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    feed: str = "iex"  # "iex" is free on Alpaca; "sip" needs a paid data subscription
    timeframe_hours: int = 1
    lookback_days: int = 180

    # Risk controls
    risk_per_trade: float = 0.01      # losing the stop costs about 1% of equity
    max_position_pct: float = 0.20    # never more than 20% of equity in one name
    max_open_positions: int = 3
    max_new_entries_per_cycle: int = 1
    max_daily_loss_pct: float = 0.02  # no new entries after a 2% drop vs. last close
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0

    # Strategy selection, judged only on held-out data
    oos_fraction: float = 0.30
    min_oos_trades: int = 5
    min_oos_sharpe: float = 0.5

    # Backtest costs
    slippage_bps: float = 5.0
    commission_bps: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        raw_universe = os.getenv("TRADER_UNIVERSE", "")
        universe = tuple(s.strip().upper() for s in raw_universe.split(",") if s.strip())
        return cls(
            api_key=os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY", ""),
            api_secret=os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET", ""),
            universe=universe or DEFAULT_UNIVERSE,
            feed=os.getenv("TRADER_FEED", "iex"),
            timeframe_hours=_int("TRADER_TIMEFRAME_HOURS", 1),
            lookback_days=_int("TRADER_LOOKBACK_DAYS", 180),
            risk_per_trade=_float("TRADER_RISK_PER_TRADE", 0.01),
            max_position_pct=_float("TRADER_MAX_POSITION_PCT", 0.20),
            max_open_positions=_int("TRADER_MAX_OPEN_POSITIONS", 3),
            max_daily_loss_pct=_float("TRADER_MAX_DAILY_LOSS_PCT", 0.02),
            atr_stop_mult=_float("TRADER_ATR_STOP_MULT", 2.0),
            atr_target_mult=_float("TRADER_ATR_TARGET_MULT", 3.0),
            min_oos_trades=_int("TRADER_MIN_OOS_TRADES", 5),
            min_oos_sharpe=_float("TRADER_MIN_OOS_SHARPE", 0.5),
            slippage_bps=_float("TRADER_SLIPPAGE_BPS", 5.0),
        )

    def require_credentials(self) -> None:
        if not (self.api_key and self.api_secret):
            raise SystemExit("Missing ALPACA_API_KEY / ALPACA_API_SECRET (use paper-trading keys).")
