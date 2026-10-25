"""Thin wrapper around alpaca-py's trading client. All order handling lives here.

Paper trading is hard-coded. Going live needs a deliberate code change, not an env var.
"""
from __future__ import annotations

from datetime import datetime

from .risk import EntryPlan


class AlpacaBroker:
    def __init__(self, api_key: str, api_secret: str):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.trading = TradingClient(api_key, api_secret, paper=True)
        self.data = StockHistoricalDataClient(api_key, api_secret)

    def account(self):
        return self.trading.get_account()

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.trading.get_all_positions()}

    def has_order_since(self, symbol: str, since: datetime) -> bool:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[symbol], after=since, limit=50)
        return len(self.trading.get_orders(filter=request)) > 0

    def submit_bracket_buy(self, symbol: str, plan: EntryPlan, client_order_id: str):
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        # GTC so the stop and target still protect the position overnight.
        request = MarketOrderRequest(
            symbol=symbol,
            qty=plan.qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=plan.target_price),
            stop_loss=StopLossRequest(stop_price=plan.stop_price),
            client_order_id=client_order_id,
        )
        return self.trading.submit_order(order_data=request)

    def close_long(self, symbol: str):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        # Cancel the open stop and target legs first, or they could reopen the position.
        open_orders = self.trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        for order in open_orders:
            self.trading.cancel_order_by_id(order.id)
        return self.trading.close_position(symbol)
