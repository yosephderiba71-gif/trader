"""Alpaca execution for the frozen multi-asset momentum rules.

Current state always comes from Alpaca's positions endpoint
(``api.list_positions()`` in the REST API, ``get_all_positions()`` in
alpaca-py). No flat file is used as position state.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

CASH = "CASH"
_EPS = 1e-9


@dataclass(frozen=True)
class RebalanceOrder:
    symbol: str
    side: str
    qty: float
    current_qty: float
    target_qty: float


def compute_rebalance_orders(
    equity: float,
    targets: Mapping[str, float],
    positions: Mapping[str, float],
    prices: Mapping[str, float],
) -> list[RebalanceOrder]:
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be positive")

    tradable: dict[str, float] = {}
    for symbol, weight in targets.items():
        if symbol == CASH:
            continue
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid target weight for {symbol}: {weight}")
        if weight > 0:
            tradable[symbol] = weight
    if sum(tradable.values()) > 1.0 + 1e-6:
        raise ValueError("target weights sum to more than 100%")

    exits: list[RebalanceOrder] = []
    for symbol, qty in positions.items():
        qty = float(qty)
        if symbol == CASH or abs(qty) < _EPS:
            continue
        if symbol not in tradable:
            side = "sell" if qty > 0 else "buy"
            exits.append(RebalanceOrder(symbol, side, abs(qty), qty, 0.0))

    trims: list[RebalanceOrder] = []
    buys: list[RebalanceOrder] = []
    for symbol, weight in tradable.items():
        price = prices.get(symbol)
        if price is None or not np.isfinite(price) or float(price) <= 0:
            raise ValueError(f"missing or invalid price for {symbol}")
        target_qty = float(math.floor(weight * equity / float(price) + _EPS))
        current_qty = float(positions.get(symbol, 0.0))
        delta = target_qty - current_qty
        if delta < -_EPS:
            trims.append(RebalanceOrder(symbol, "sell", abs(delta), current_qty, target_qty))
        elif delta > _EPS:
            buys.append(RebalanceOrder(symbol, "buy", delta, current_qty, target_qty))

    exits.sort(key=lambda order: order.symbol)
    trims.sort(key=lambda order: order.symbol)
    buys.sort(key=lambda order: order.symbol)
    return exits + trims + buys


def _credentials() -> tuple[str, str]:
    api_key = (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("ALPACA_KEY")
    )
    api_secret = (
        os.environ.get("ALPACA_API_SECRET")
        or os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET")
    )
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET "
            "in the environment."
        )
    return api_key, api_secret


def _paper_mode() -> bool:
    raw = os.environ.get("ALPACA_PAPER", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _build_clients():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    api_key, api_secret = _credentials()
    trading = TradingClient(
        api_key,
        api_secret,
        paper=_paper_mode(),
        url_override=os.environ.get("ALPACA_BASE_URL") or None,
    )
    data = StockHistoricalDataClient(api_key, api_secret)
    return trading, data


def _fetch_latest_prices(data_client, symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestTradeRequest

    feed_name = os.environ.get("ALPACA_DATA_FEED", "iex").strip().lower()
    try:
        feed = DataFeed(feed_name)
    except ValueError:
        feed = DataFeed.IEX
    request = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=feed)
    trades = data_client.get_stock_latest_trade(request)
    return {symbol: float(trade.price) for symbol, trade in trades.items()}


def _print_plan(
    equity: float,
    positions: Mapping[str, float],
    targets: Mapping[str, float],
    orders: list[RebalanceOrder],
) -> None:
    print(f"Account equity: ${equity:,.2f}")
    held = ", ".join(f"{symbol} {qty:g}" for symbol, qty in sorted(positions.items())) or "none"
    print(f"Current positions: {held}")
    print(
        "Target weights: "
        + ", ".join(f"{symbol} {weight:.2%}" for symbol, weight in targets.items())
    )
    if not orders:
        print("No trades required.")
        return
    for order in orders:
        print(
            f"  {order.side.upper():<4} {order.qty:g} {order.symbol} "
            f"(current {order.current_qty:g} -> target {order.target_qty:g})"
        )


def _submit_orders(trading_client, orders: list[RebalanceOrder]) -> None:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    for order in orders:
        qty = int(order.qty) if float(order.qty).is_integer() else order.qty
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=qty,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        submitted = trading_client.submit_order(order_data=request)
        print(
            f"Submitted {order.side} {order.qty:g} {order.symbol}: "
            f"id={submitted.id} status={submitted.status}"
        )


def rebalance(
    targets: Mapping[str, float],
    latest_prices: Mapping[str, float] | None = None,
    dry_run: bool = True,
    trading_client=None,
    data_client=None,
) -> list[RebalanceOrder]:
    if trading_client is None:
        trading_client, data_client = _build_clients()
    elif data_client is None:
        _, data_client = _build_clients()

    account = trading_client.get_account()
    equity = float(account.equity)
    raw_positions = trading_client.get_all_positions()
    positions = {position.symbol: float(position.qty) for position in raw_positions}

    prices: dict[str, float] = {}
    for position in raw_positions:
        current_price = getattr(position, "current_price", None)
        if current_price is None:
            continue
        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            continue
        if current_price > 0:
            prices[position.symbol] = current_price

    if latest_prices:
        prices.update(
            {
                symbol: float(price)
                for symbol, price in latest_prices.items()
                if price is not None and float(price) > 0
            }
        )

    needed = [
        symbol
        for symbol, weight in targets.items()
        if symbol != CASH and weight > 0 and symbol not in prices
    ]
    if needed:
        prices.update(_fetch_latest_prices(data_client, needed))

    orders = compute_rebalance_orders(equity, targets, positions, prices)
    _print_plan(equity, positions, targets, orders)
    if dry_run:
        print("DRY RUN: no orders submitted.")
        return orders
    _submit_orders(trading_client, orders)
    return orders
