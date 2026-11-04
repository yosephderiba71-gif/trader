"""Thin Alpaca wrapper for the intraday sleeve. Paper trading is hard-coded."""
from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


class AlpacaBroker:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.trading = TradingClient(api_key, api_secret, paper=paper)
        self.data = StockHistoricalDataClient(api_key, api_secret)

    def clock(self):
        return self.trading.get_clock()

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def account(self):
        return self.trading.get_account()

    def position(self, symbol: str):
        try:
            return self.trading.get_open_position(symbol)
        except Exception:
            return None

    def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        feed: str = "iex",
        bar_minutes: int = 5,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        now = now or datetime.now(timezone.utc)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(bar_minutes, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=DataFeed(feed),
        )
        raw = self.data.get_stock_bars(request).df
        if raw.empty:
            return pd.DataFrame(columns=OHLCV)
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(symbol, level=0)
        frame = raw[OHLCV].sort_index()
        if not frame.empty and frame.index[-1] + pd.Timedelta(minutes=bar_minutes) > pd.Timestamp(now):
            frame = frame.iloc[:-1]
        return frame

    def submit_market(self, symbol: str, qty: int, side: str, client_order_id: str):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        return self.trading.submit_order(order_data=request)

    def wait_for_fill(self, order_id: str, timeout: float = 15.0):
        from alpaca.trading.enums import OrderStatus

        deadline = _time.monotonic() + timeout
        order = None
        while _time.monotonic() < deadline:
            order = self.trading.get_order_by_id(order_id)
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}:
                return order
            _time.sleep(1.0)
        return order

    def cancel_open_orders(self, symbol: str) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.trading.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], limit=100)
        )
        for order in orders:
            try:
                self.trading.cancel_order_by_id(order.id)
            except Exception:
                pass

    def close_position(self, symbol: str):
        self.cancel_open_orders(symbol)
        return self.trading.close_position(symbol)

    def orders_since(self, symbol: str, after: datetime) -> list:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return list(
            self.trading.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.ALL,
                    symbols=[symbol],
                    after=after,
                    limit=100,
                )
            )
        )


def fetch_window(now: datetime | None = None, sessions: int = 20) -> tuple[datetime, datetime]:
    """A generous UTC window that covers enough completed regular sessions."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=max(sessions * 2, 40))
    return start, now
