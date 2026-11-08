"""Unit tests for the Discipline Bot (Cash-Flow Rebalancing & Dip Accelerator)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from discipline_bot.execute import (
    BASE_DEPOSIT,
    SPY_TARGET,
    TLT_TARGET,
    assert_paper_mode,
    calculate_deposit_amount,
    calculate_dip_multiplier,
    format_client_order_id,
    has_filled_this_month,
    plan_rebalance,
    run_discipline_cycle,
)


def test_dip_accelerator_scaling():
    """Verify deposit scales correctly based on SPY 52-week high drawdown."""
    high_52w = 100.0

    # 1. Normal market: SPY down 5% (not >15% down) -> 1.0x = $500
    mult_normal = calculate_dip_multiplier(current_price=95.0, high_52w=high_52w)
    assert mult_normal == 1.0
    assert calculate_deposit_amount(500.0, 95.0, high_52w) == 500.0

    # 2. Down >15% (e.g. 16% down: price 84.0) -> 1.25x = $625
    mult_15 = calculate_dip_multiplier(current_price=84.0, high_52w=high_52w)
    assert mult_15 == 1.25
    assert calculate_deposit_amount(500.0, 84.0, high_52w) == 625.0

    # Boundary: exactly 15% down -> 1.0x (rule: >15%)
    assert calculate_dip_multiplier(current_price=85.0, high_52w=high_52w) == 1.0

    # 3. Down >25% (e.g. 26% down: price 74.0) -> 1.50x = $750
    mult_25 = calculate_dip_multiplier(current_price=74.0, high_52w=high_52w)
    assert mult_25 == 1.50
    assert calculate_deposit_amount(500.0, 74.0, high_52w) == 750.0

    # Boundary: exactly 25% down -> 1.25x (rule: >25%)
    assert calculate_dip_multiplier(current_price=75.0, high_52w=high_52w) == 1.25


def test_cash_flow_rebalance_allocates_to_underweight_tlt():
    """When TLT weight drops below 17%, 100% of new cash is directed to TLT."""
    deposit = 500.0
    plan = plan_rebalance(spy_value=85000.0, tlt_value=15000.0, deposit=deposit)

    # 100% of new cash goes to TLT
    assert len(plan.buy_orders) == 1
    assert plan.buy_orders[0].symbol == "TLT"
    assert plan.buy_orders[0].notional == pytest.approx(500.0)

    # SPY > 83% and new cash is insufficient, triggering an explicit sell rebalance
    assert plan.requires_sell_rebalance is True
    assert len(plan.sell_orders) == 1
    assert plan.sell_orders[0].symbol == "SPY"


def test_cash_flow_rebalance_sufficient_cash_no_sell():
    """When new cash is sufficient to fix drift within band, no sell orders are created."""
    plan = plan_rebalance(spy_value=8100.0, tlt_value=1900.0, deposit=500.0)

    assert plan.requires_sell_rebalance is False
    assert len(plan.sell_orders) == 0
    assert len(plan.buy_orders) == 1
    assert plan.buy_orders[0].symbol == "TLT"
    assert plan.buy_orders[0].notional == pytest.approx(500.0)


def test_cash_flow_rebalance_allocates_to_underweight_spy():
    """When SPY weight is underweight, 100% of new cash is directed to SPY."""
    plan = plan_rebalance(spy_value=7500.0, tlt_value=2500.0, deposit=500.0)

    assert len(plan.buy_orders) == 1
    assert plan.buy_orders[0].symbol == "SPY"
    assert plan.buy_orders[0].notional == pytest.approx(500.0)


def test_calendar_guard_aborts_on_8th_or_later():
    """The bot aborts if the current day is the 8th of the month or later."""
    # Day 8: must abort
    day_8 = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    res_8 = run_discipline_cycle(current_date=day_8, paper=True)
    assert res_8["status"] == "SKIPPED_CALENDAR_GUARD"
    assert res_8["orders_submitted"] == []

    # Day 15: must abort
    day_15 = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    res_15 = run_discipline_cycle(current_date=day_15, paper=True)
    assert res_15["status"] == "SKIPPED_CALENDAR_GUARD"
    assert res_15["orders_submitted"] == []

    # Day 7: first week boundary, must NOT trigger calendar guard
    mock_client = MagicMock()
    mock_client.get_activities.return_value = []
    mock_client.get_account.return_value = SimpleNamespace(equity="10000.0")
    day_7 = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    res_7 = run_discipline_cycle(
        trading_client=mock_client,
        current_date=day_7,
        paper=True,
        dry_run=True,
        mock_prices={"SPY": 500.0, "TLT": 90.0, "SPY_52W_HIGH": 520.0},
        mock_positions={"SPY": 16.0, "TLT": 22.0},
    )
    assert res_7["status"] == "DRY_RUN"


def test_client_order_id_formatting():
    """The bot correctly formats the client_order_id as discipline_YYYYMM."""
    dt_nov = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    assert format_client_order_id(dt_nov) == "discipline_202611"

    dt_jan = datetime(2027, 1, 5, 12, 0, tzinfo=timezone.utc)
    assert format_client_order_id(dt_jan) == "discipline_202701"

    # Orders planned for November 2026 carry the expected client_order_id
    plan = plan_rebalance(
        spy_value=8000.0,
        tlt_value=2000.0,
        deposit=500.0,
        client_order_id=format_client_order_id(dt_nov),
    )
    assert plan.buy_orders[0].client_order_id == "discipline_202611"


def test_strict_idempotency_matching():
    """The idempotency check ONLY triggers on matching client_order_ids and ignores others."""
    current_dt = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)

    # 1. Exact match for this month's discipline bot order -> ABORT (True)
    matching_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="discipline_202611",
            symbol="TLT",
        )
    ]
    assert has_filled_this_month(matching_fill, current_date=current_dt) is True

    # Also works with dict format returned by Alpaca REST API
    matching_dict = [
        {
            "activity_type": "FILL",
            "client_order_id": "discipline_202611",
            "symbol": "TLT",
        }
    ]
    assert has_filled_this_month(matching_dict, current_date=current_dt) is True

    # 2. Manual trade with different client_order_id -> IGNORED (False)
    manual_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="manual_order_12345",
            symbol="SPY",
        )
    ]
    assert has_filled_this_month(manual_fill, current_date=current_dt) is False

    # 3. Fill with no client_order_id -> IGNORED (False)
    none_cid_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id=None,
            symbol="SPY",
        )
    ]
    assert has_filled_this_month(none_cid_fill, current_date=current_dt) is False

    # 4. Dividend activity -> IGNORED (False)
    dividend_activity = [
        SimpleNamespace(
            activity_type="DIV",
            client_order_id="discipline_202611",
            symbol="SPY",
        )
    ]
    assert has_filled_this_month(dividend_activity, current_date=current_dt) is False

    # 5. Discipline fill from prior month -> IGNORED (False)
    prior_month_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="discipline_202610",
            symbol="TLT",
        )
    ]
    assert has_filled_this_month(prior_month_fill, current_date=current_dt) is False


def test_hard_paper_assertion():
    """Raise ValueError if paper=True is not explicitly set."""
    with pytest.raises(ValueError, match="Hard paper assertion failed"):
        assert_paper_mode(paper=False)

    assert_paper_mode(paper=True)


def test_idempotency_cycle_abort():
    """Execution cycle aborts without submitting orders if matching fill occurred this month."""
    mock_client = MagicMock()
    mock_client.get_activities.return_value = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="discipline_202611",
            symbol="TLT",
        )
    ]
    mock_client.get_account.return_value = SimpleNamespace(equity="10000.0")

    result = run_discipline_cycle(
        trading_client=mock_client,
        paper=True,
        dry_run=False,
        base_deposit=500.0,
        current_date=datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc),  # within first week
        mock_prices={"SPY": 500.0, "TLT": 90.0, "SPY_52W_HIGH": 520.0},
        mock_positions={"SPY": 16.0, "TLT": 22.0},
    )

    assert result["status"] == "ABORTED_ALREADY_FILLED"
    assert result["orders_submitted"] == []
    mock_client.submit_order.assert_not_called()
