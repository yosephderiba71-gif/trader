"""Discipline Bot: Cash-Flow Rebalancing with Dip Accelerator.

Specifications (research/discipline_bot_rules.md):
- Assets: SPY (80%), TLT (20%).
- Execution: Standard Market Orders (fractional shares via `notional`) at 12:00 EST.
- Cash-Flow Rebalancing: Target Drift Band ±3% (SPY 77%-83%, TLT 17%-23%).
  Direct new cash 100% into the asset that is currently underweight.
  Explicit Sell Rebalance: Only trigger if an asset breaches outer band (e.g. SPY > 83%)
  AND new cash is insufficient to fix it.
- Market Dip Accelerator:
  Base Deposit: $500 (configurable).
  If SPY is >15% below 52-week high: Deposit = $625 (1.25x).
  If SPY is >25% below 52-week high: Deposit = $750 (1.50x).
- Production Hardening & Safety:
  1. Calendar Guard: Hard check at the very beginning of execution.
     If day > 7, print 'Not the first week of the month. Skipping.' and return.
  2. Client Order ID: Tag order with client_order_id=f"discipline_{YYYYMM}".
  3. Strict Idempotency: Query Alpaca Activity API (`get_activities`).
     ONLY match filled orders where client_order_id exactly matches discipline_{YYYYMM}.
     Ignore manual trades and dividends.
  4. Hard Paper Assertion: Raise ValueError if `paper=True` is not explicitly set.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

SPY_TARGET = 0.80
TLT_TARGET = 0.20
DRIFT_BAND = 0.03
SPY_MIN_WEIGHT = SPY_TARGET - DRIFT_BAND  # 0.77
SPY_MAX_WEIGHT = SPY_TARGET + DRIFT_BAND  # 0.83
TLT_MIN_WEIGHT = TLT_TARGET - DRIFT_BAND  # 0.17
TLT_MAX_WEIGHT = TLT_TARGET + DRIFT_BAND  # 0.23

BASE_DEPOSIT = 500.0
DIP_15_THRESHOLD = 0.15
DIP_25_THRESHOLD = 0.25
DIP_15_MULTIPLIER = 1.25
DIP_25_MULTIPLIER = 1.50


@dataclass(frozen=True)
class BotOrder:
    symbol: str
    side: str  # "buy" or "sell"
    notional: float
    client_order_id: str | None = None


@dataclass
class RebalancePlan:
    spy_value: float
    tlt_value: float
    total_equity: float
    spy_weight: float
    tlt_weight: float
    underweight_asset: str
    deposit: float
    requires_sell_rebalance: bool
    buy_orders: list[BotOrder]
    sell_orders: list[BotOrder]


def format_client_order_id(current_date: date | datetime | None = None) -> str:
    """Format client_order_id as discipline_YYYYMM."""
    current_date = current_date or datetime.now()
    return f"discipline_{current_date.strftime('%Y%m')}"


def assert_paper_mode(paper: bool) -> None:
    """Hard paper assertion: raise ValueError if paper is not True."""
    if not paper:
        raise ValueError(
            "Hard paper assertion failed: paper=True is required. Live production trading is strictly disallowed."
        )


def calculate_dip_multiplier(current_price: float, high_52w: float) -> float:
    """Calculate deposit multiplier based on drawdown from 52-week high."""
    if not math.isfinite(current_price) or not math.isfinite(high_52w) or high_52w <= 0:
        return 1.0
    drawdown = (current_price - high_52w) / high_52w
    # Rule: If SPY is >25% below its 52-week high: 1.50x
    if drawdown < -DIP_25_THRESHOLD:
        return DIP_25_MULTIPLIER
    # Rule: If SPY is >15% below its 52-week high: 1.25x
    if drawdown < -DIP_15_THRESHOLD:
        return DIP_15_MULTIPLIER
    return 1.0


def calculate_deposit_amount(base_deposit: float, current_price: float, high_52w: float) -> float:
    """Calculate final dollar deposit amount to deploy."""
    multiplier = calculate_dip_multiplier(current_price, high_52w)
    return round(base_deposit * multiplier, 2)


def plan_rebalance(
    spy_value: float,
    tlt_value: float,
    deposit: float,
    client_order_id: str | None = None,
) -> RebalancePlan:
    """Compute cash-flow rebalance allocation and check if sell rebalance is needed."""
    total_assets = spy_value + tlt_value
    cid = client_order_id or format_client_order_id()

    if total_assets <= 0:
        underweight = "SPY"
        return RebalancePlan(
            spy_value=0.0,
            tlt_value=0.0,
            total_equity=0.0,
            spy_weight=0.0,
            tlt_weight=0.0,
            underweight_asset=underweight,
            deposit=deposit,
            requires_sell_rebalance=False,
            buy_orders=[BotOrder(symbol=underweight, side="buy", notional=deposit, client_order_id=cid)],
            sell_orders=[],
        )

    spy_weight = spy_value / total_assets
    tlt_weight = tlt_value / total_assets

    spy_dev = spy_weight - SPY_TARGET
    tlt_dev = tlt_weight - TLT_TARGET

    if tlt_dev < spy_dev:
        underweight = "TLT"
        hypo_total = total_assets + deposit
        hypo_spy_weight = spy_value / hypo_total
        buy_orders = [BotOrder(symbol="TLT", side="buy", notional=deposit, client_order_id=cid)]

        if hypo_spy_weight > SPY_MAX_WEIGHT:
            requires_sell = True
            target_spy = SPY_TARGET * hypo_total
            sell_notional = round(spy_value - target_spy, 2)
            sell_orders = [BotOrder(symbol="SPY", side="sell", notional=sell_notional, client_order_id=f"{cid}_sell")]
        else:
            requires_sell = False
            sell_orders = []
    else:
        underweight = "SPY"
        hypo_total = total_assets + deposit
        hypo_tlt_weight = tlt_value / hypo_total
        buy_orders = [BotOrder(symbol="SPY", side="buy", notional=deposit, client_order_id=cid)]

        if hypo_tlt_weight > TLT_MAX_WEIGHT:
            requires_sell = True
            target_tlt = TLT_TARGET * hypo_total
            sell_notional = round(tlt_value - target_tlt, 2)
            sell_orders = [BotOrder(symbol="TLT", side="sell", notional=sell_notional, client_order_id=f"{cid}_sell")]
        else:
            requires_sell = False
            sell_orders = []

    return RebalancePlan(
        spy_value=spy_value,
        tlt_value=tlt_value,
        total_equity=total_assets,
        spy_weight=spy_weight,
        tlt_weight=tlt_weight,
        underweight_asset=underweight,
        deposit=deposit,
        requires_sell_rebalance=requires_sell,
        buy_orders=buy_orders,
        sell_orders=sell_orders,
    )


def has_filled_this_month(
    activities: list[Any],
    current_date: date | datetime | None = None,
) -> bool:
    """Strict idempotency check.
    
    ONLY looks for filled orders where client_order_id exactly matches
    discipline_{current_year_month}. Ignores manual trades or dividends.
    """
    current_date = current_date or datetime.now()
    expected_client_order_id = format_client_order_id(current_date)

    for act in activities:
        act_type = (
            act.get("activity_type")
            if isinstance(act, dict)
            else getattr(act, "activity_type", None)
        )
        # Ignore dividends, corporate actions, non-fill activities
        if str(act_type).upper() != "FILL":
            continue

        cid = (
            act.get("client_order_id")
            if isinstance(act, dict)
            else getattr(act, "client_order_id", None)
        )
        # Strict match: only abort if client_order_id matches discipline_{YYYYMM}
        if cid == expected_client_order_id:
            return True

    return False


def _load_env_file(path: str = ".env") -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    if k not in os.environ:
                        os.environ[k] = v


def _get_alpaca_credentials() -> tuple[str, str]:
    _load_env_file()
    key = (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("ALPACA_KEY")
    )
    secret = (
        os.environ.get("ALPACA_API_SECRET")
        or os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET")
    )
    if not key or not secret:
        raise RuntimeError("Missing Alpaca API credentials. Please set ALPACA_API_KEY and ALPACA_API_SECRET.")
    return key, secret


def build_trading_client(paper: bool = True):
    from alpaca.trading.client import TradingClient

    assert_paper_mode(paper)
    key, secret = _get_alpaca_credentials()
    return TradingClient(key, secret, paper=True)


def get_account_activities(trading_client) -> list[Any]:
    if hasattr(trading_client, "get_activities"):
        return trading_client.get_activities(activity_type="FILL")
    try:
        return trading_client.get("/account/activities/FILL")
    except Exception:
        return trading_client.get("/account/activities")


def fetch_market_data() -> dict[str, float]:
    import yfinance as yf

    spy_ticker = yf.Ticker("SPY")
    tlt_ticker = yf.Ticker("TLT")

    spy_hist = spy_ticker.history(period="1y", interval="1d", auto_adjust=True)
    tlt_hist = tlt_ticker.history(period="1mo", interval="1d", auto_adjust=True)

    if spy_hist is None or spy_hist.empty or "Close" not in spy_hist.columns:
        raise RuntimeError("Failed to fetch SPY price data from yfinance.")
    if tlt_hist is None or tlt_hist.empty or "Close" not in tlt_hist.columns:
        raise RuntimeError("Failed to fetch TLT price data from yfinance.")

    spy_close = float(spy_hist["Close"].dropna().iloc[-1])
    spy_high = float(spy_hist["High"].dropna().max())
    tlt_close = float(tlt_hist["Close"].dropna().iloc[-1])

    return {
        "SPY": spy_close,
        "SPY_52W_HIGH": spy_high,
        "TLT": tlt_close,
    }


def run_discipline_cycle(
    trading_client=None,
    paper: bool = True,
    dry_run: bool = True,
    base_deposit: float = BASE_DEPOSIT,
    current_date: date | datetime | None = None,
    mock_prices: dict[str, float] | None = None,
    mock_positions: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Execute one full Discipline Bot cycle."""
    # 1. Calendar Guard (hard check at the very beginning)
    now = current_date or datetime.now()
    if now.day > 7:
        print("Not the first week of the month. Skipping.")
        return {
            "status": "SKIPPED_CALENDAR_GUARD",
            "reason": "Not the first week of the month. Skipping.",
            "orders_submitted": [],
        }

    # Hard Paper Assertion
    assert_paper_mode(paper)

    if trading_client is None:
        trading_client = build_trading_client(paper=paper)

    # 2. Strict Idempotency Check
    activities = get_account_activities(trading_client)
    if has_filled_this_month(activities, current_date=now):
        print(f"Idempotency guard: An order with client_order_id='{format_client_order_id(now)}' has already filled this month.")
        print("Cycle aborted. No orders will be placed.")
        return {
            "status": "ABORTED_ALREADY_FILLED",
            "reason": f"Order already filled with client_order_id={format_client_order_id(now)}",
            "orders_submitted": [],
        }

    # 3. Market Data & 52-Week High Drawdown
    prices = mock_prices or fetch_market_data()
    spy_close = prices["SPY"]
    spy_high = prices["SPY_52W_HIGH"]
    tlt_close = prices["TLT"]

    drawdown_pct = (spy_close - spy_high) / spy_high
    multiplier = calculate_dip_multiplier(spy_close, spy_high)
    deposit = calculate_deposit_amount(base_deposit, spy_close, spy_high)

    # 4. Account Equity & Current Positions
    account = trading_client.get_account()
    equity = float(account.equity)

    if mock_positions is not None:
        spy_qty = float(mock_positions.get("SPY", 0.0))
        tlt_qty = float(mock_positions.get("TLT", 0.0))
        spy_value = spy_qty * spy_close
        tlt_value = tlt_qty * tlt_close
    else:
        positions = trading_client.get_all_positions()
        pos_by_sym = {p.symbol: p for p in positions}
        spy_pos = pos_by_sym.get("SPY")
        tlt_pos = pos_by_sym.get("TLT")
        spy_value = float(getattr(spy_pos, "market_value", 0.0) or 0.0)
        tlt_value = float(getattr(tlt_pos, "market_value", 0.0) or 0.0)

    # 5. Plan Rebalance with client_order_id
    expected_cid = format_client_order_id(now)
    plan = plan_rebalance(
        spy_value=spy_value,
        tlt_value=tlt_value,
        deposit=deposit,
        client_order_id=expected_cid,
    )

    # Print Summary Report
    print("=" * 65)
    print("Discipline Bot: Cash-Flow Rebalancing & Dip Accelerator")
    print("=" * 65)
    print(f"Mode: {'DRY RUN (no orders submitted)' if dry_run else 'EXECUTE (live paper orders)'}")
    print(f"Account Equity: ${equity:,.2f}")
    print(f"SPY Position:   ${spy_value:,.2f} ({plan.spy_weight:.2%}) [Target: 80% ±3%]")
    print(f"TLT Position:   ${tlt_value:,.2f} ({plan.tlt_weight:.2%}) [Target: 20% ±3%]")
    print("-" * 65)
    print(f"SPY Current Price:   ${spy_close:.2f}")
    print(f"SPY 52-Week High:    ${spy_high:.2f}")
    print(f"Drawdown from High:  {drawdown_pct:.2%}")
    print(f"Dip Multiplier:      {multiplier:.2f}x (Base: ${base_deposit:.2f})")
    print(f"Deposit to Deploy:   ${deposit:.2f}")
    print(f"Underweight Asset:   {plan.underweight_asset}")
    print(f"Client Order ID:     {expected_cid}")
    print("-" * 65)
    print("Planned Orders:")
    for order in plan.buy_orders:
        print(f"  BUY  {order.symbol:<4} ${order.notional:,.2f} (client_order_id={order.client_order_id})")
    for order in plan.sell_orders:
        print(f"  SELL {order.symbol:<4} ${order.notional:,.2f} (client_order_id={order.client_order_id})")
    if not plan.sell_orders and not plan.buy_orders:
        print("  None.")
    print("=" * 65)

    if dry_run:
        return {
            "status": "DRY_RUN",
            "plan": plan,
            "orders_submitted": [],
            "deposit": deposit,
            "multiplier": multiplier,
            "client_order_id": expected_cid,
        }

    # 6. Order Execution with Client Order ID
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    submitted_orders = []
    for order in plan.sell_orders:
        req = MarketOrderRequest(
            symbol=order.symbol,
            notional=round(order.notional, 2),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.client_order_id or f"{expected_cid}_sell",
        )
        sub = trading_client.submit_order(req)
        submitted_orders.append(sub)
        print(f"Submitted SELL order for {order.symbol}: notional=${order.notional:.2f}, client_order_id={req.client_order_id}, id={sub.id}")

    for order in plan.buy_orders:
        req = MarketOrderRequest(
            symbol=order.symbol,
            notional=round(order.notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.client_order_id or expected_cid,
        )
        sub = trading_client.submit_order(req)
        submitted_orders.append(sub)
        print(f"Submitted BUY order for {order.symbol}: notional=${order.notional:.2f}, client_order_id={req.client_order_id}, id={sub.id}")

    return {
        "status": "SUCCESS",
        "plan": plan,
        "orders_submitted": submitted_orders,
        "deposit": deposit,
        "multiplier": multiplier,
        "client_order_id": expected_cid,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discipline Bot: Cash-Flow Rebalancing with Dip Accelerator."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute paper orders on Alpaca (default is dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode without submitting orders (default: True).",
    )
    parser.add_argument(
        "--base-deposit",
        type=float,
        default=BASE_DEPOSIT,
        help=f"Base monthly deposit amount in USD (default: {BASE_DEPOSIT}).",
    )

    args = parser.parse_args(argv)
    is_dry_run = not args.execute

    run_discipline_cycle(
        paper=True,
        dry_run=is_dry_run,
        base_deposit=args.base_deposit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
