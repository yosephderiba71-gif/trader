"""Unit tests for the Discipline Bot (Cash-Flow Rebalancing & Dip Accelerator)."""
from __future__ import annotations

from datetime import datetime, timezone
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
    eastern_now,
    format_client_order_id,
    get_existing_orders,
    has_existing_week_order,
    has_filled_this_week,
    plan_rebalance,
    run_discipline_cycle,
)

MARKET_PRICES = {"SPY": 500.0, "TLT": 90.0, "SPY_52W_HIGH": 520.0}
POSITIONS = {"SPY": 16.0, "TLT": 22.0}


def make_client(
    *,
    cash="10000.0",
    buying_power="20000.0",
    non_marginable_buying_power=None,
    equity="10000.0",
    is_open=True,
    clock_timestamp=None,
    activities=None,
    orders=None,
):
    client = MagicMock()
    client.get_clock.return_value = SimpleNamespace(
        is_open=is_open,
        timestamp=clock_timestamp or datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
    )
    client.get_activities.return_value = activities if activities is not None else []
    client.get_orders.return_value = orders if orders is not None else []

    account_fields = {
        "equity": equity,
        "cash": cash,
        "buying_power": buying_power,
    }
    if non_marginable_buying_power is not None:
        account_fields["non_marginable_buying_power"] = non_marginable_buying_power
    client.get_account.return_value = SimpleNamespace(**account_fields)
    client.submit_order.return_value = SimpleNamespace(id="order-123")
    return client


def test_dip_accelerator_scaling():
    """Verify deposit scales correctly based on SPY 52-week high drawdown."""
    high_52w = 100.0
    assert BASE_DEPOSIT == 125.0

    # 1. Normal market: SPY down 5% (not >15% down) -> 1.0x = $125
    mult_normal = calculate_dip_multiplier(current_price=95.0, high_52w=high_52w)
    assert mult_normal == 1.0
    assert calculate_deposit_amount(BASE_DEPOSIT, 95.0, high_52w) == 125.0

    # 2. Down >15% (e.g. 16% down: price 84.0) -> 1.25x = $156.25
    mult_15 = calculate_dip_multiplier(current_price=84.0, high_52w=high_52w)
    assert mult_15 == 1.25
    assert calculate_deposit_amount(BASE_DEPOSIT, 84.0, high_52w) == 156.25

    # Boundary: exactly 15% down -> 1.0x (rule: >15%)
    assert calculate_dip_multiplier(current_price=85.0, high_52w=high_52w) == 1.0

    # 3. Down >25% (e.g. 26% down: price 74.0) -> 1.50x = $187.50
    mult_25 = calculate_dip_multiplier(current_price=74.0, high_52w=high_52w)
    assert mult_25 == 1.50
    assert calculate_deposit_amount(BASE_DEPOSIT, 74.0, high_52w) == 187.50

    # Boundary: exactly 25% down -> 1.25x (rule: >25%)
    assert calculate_dip_multiplier(current_price=75.0, high_52w=high_52w) == 1.25


def test_cash_flow_rebalance_allocates_to_underweight_tlt():
    """When TLT weight drops below 17%, 100% of new cash is directed to TLT."""
    plan = plan_rebalance(spy_value=85000.0, tlt_value=15000.0, deposit=BASE_DEPOSIT)

    # 100% of new cash goes to TLT
    assert len(plan.buy_orders) == 1
    assert plan.buy_orders[0].symbol == "TLT"
    assert plan.buy_orders[0].notional == pytest.approx(BASE_DEPOSIT)

    # SPY > 83% and new cash is insufficient, triggering an explicit sell rebalance
    assert plan.requires_sell_rebalance is True
    assert len(plan.sell_orders) == 1
    assert plan.sell_orders[0].symbol == "SPY"


def test_cash_flow_rebalance_sufficient_cash_no_sell():
    """When new cash is sufficient to fix drift within band, no sell orders are created."""
    plan = plan_rebalance(spy_value=8100.0, tlt_value=1900.0, deposit=BASE_DEPOSIT)

    assert plan.requires_sell_rebalance is False
    assert len(plan.sell_orders) == 0
    assert len(plan.buy_orders) == 1
    assert plan.buy_orders[0].symbol == "TLT"
    assert plan.buy_orders[0].notional == pytest.approx(BASE_DEPOSIT)


def test_cash_flow_rebalance_allocates_to_underweight_spy():
    """When SPY weight is underweight, 100% of new cash is directed to SPY."""
    plan = plan_rebalance(spy_value=7500.0, tlt_value=2500.0, deposit=BASE_DEPOSIT)

    assert len(plan.buy_orders) == 1
    assert plan.buy_orders[0].symbol == "SPY"
    assert plan.buy_orders[0].notional == pytest.approx(BASE_DEPOSIT)


def test_runs_after_first_week_when_market_is_open():
    """There is no first-week-of-month restriction; the market clock drives execution."""
    client = make_client()
    result = run_discipline_cycle(
        trading_client=client,
        current_date=datetime(2026, 11, 13, 17, 0, tzinfo=timezone.utc),
        paper=True,
        dry_run=True,
        base_deposit=BASE_DEPOSIT,
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )
    assert result["status"] == "DRY_RUN"
    assert result["orders_submitted"] == []


def test_client_order_id_formatting_is_iso_week_in_eastern_time():
    """The bot formats the client_order_id as discipline_YYYYWww (ISO week)."""
    dt_nov = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)
    assert format_client_order_id(dt_nov) == "discipline_2026W45"

    dt_jan = datetime(2027, 1, 5, 17, 0, tzinfo=timezone.utc)
    assert format_client_order_id(dt_jan) == "discipline_2027W01"

    # Orders planned for November 2026 carry the expected client_order_id
    plan = plan_rebalance(
        spy_value=8000.0,
        tlt_value=2000.0,
        deposit=BASE_DEPOSIT,
        client_order_id=format_client_order_id(dt_nov),
    )
    assert plan.buy_orders[0].client_order_id == "discipline_2026W45"


def test_late_december_iso_year_boundary_in_eastern_time():
    """2025-12-29 in America/New_York belongs to ISO year 2026, week 01."""
    late_dec = datetime(2025, 12, 29, 17, 0, tzinfo=timezone.utc)  # 12:00 ET
    assert format_client_order_id(late_dec) == "discipline_2026W01"
    assert format_client_order_id(datetime(2025, 12, 29, 12, 0)) == "discipline_2026W01"

    # 02:00 UTC on Dec 29 is still Dec 28 in ET -> ISO 2025-W52.
    pre_midnight_utc = datetime(2025, 12, 29, 2, 0, tzinfo=timezone.utc)
    assert format_client_order_id(pre_midnight_utc) == "discipline_2025W52"
    assert eastern_now(clock=SimpleNamespace(timestamp=pre_midnight_utc)).strftime("%Y-%m-%d") == "2025-12-28"

    client = make_client()
    result = run_discipline_cycle(
        trading_client=client,
        current_date=late_dec,
        paper=True,
        dry_run=True,
        base_deposit=BASE_DEPOSIT,
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )
    assert result["status"] == "DRY_RUN"
    assert result["client_order_id"] == "discipline_2026W01"

    boundary_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="discipline_2026W01",
            symbol="SPY",
        )
    ]
    assert has_filled_this_week(boundary_fill, current_date=late_dec) is True
    prior_iso_week = datetime(2025, 12, 28, 17, 0, tzinfo=timezone.utc)
    assert has_filled_this_week(boundary_fill, current_date=prior_iso_week) is False


def test_strict_weekly_idempotency_matching():
    """The idempotency check ONLY triggers on matching ISO-week client_order_ids."""
    current_dt = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)  # 2026-W45

    # 1. Exact match for this week's discipline bot order -> ABORT (True)
    matching_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="discipline_2026W45",
            symbol="TLT",
        )
    ]
    assert has_filled_this_week(matching_fill, current_date=current_dt) is True

    # Later in the same ISO week -> still True
    assert has_filled_this_week(
        matching_fill,
        current_date=datetime(2026, 11, 6, 17, 0, tzinfo=timezone.utc),
    ) is True

    # Also works with dict format returned by Alpaca REST API
    matching_dict = [
        {
            "activity_type": "FILL",
            "client_order_id": "discipline_2026W45",
            "symbol": "TLT",
        }
    ]
    assert has_filled_this_week(matching_dict, current_date=current_dt) is True

    # 2. Manual trade with different client_order_id -> IGNORED (False)
    manual_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="manual_order_12345",
            symbol="SPY",
        )
    ]
    assert has_filled_this_week(manual_fill, current_date=current_dt) is False

    # 3. Fill with no client_order_id -> IGNORED (False)
    none_cid_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id=None,
            symbol="SPY",
        )
    ]
    assert has_filled_this_week(none_cid_fill, current_date=current_dt) is False

    # 4. Dividend activity -> IGNORED (False)
    dividend_activity = [
        SimpleNamespace(
            activity_type="DIV",
            client_order_id="discipline_2026W45",
            symbol="SPY",
        )
    ]
    assert has_filled_this_week(dividend_activity, current_date=current_dt) is False

    # 5. Discipline fill from a prior ISO week -> IGNORED (False)
    prior_week_fill = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="discipline_2026W44",
            symbol="TLT",
        )
    ]
    assert has_filled_this_week(prior_week_fill, current_date=current_dt) is False


def test_hard_paper_assertion():
    """Raise ValueError if paper=True is not explicitly set."""
    with pytest.raises(ValueError, match="Hard paper assertion failed"):
        assert_paper_mode(paper=False)

    assert_paper_mode(paper=True)


def test_idempotency_cycle_abort():
    """Execution cycle aborts without submitting orders if this ISO week filled."""
    client = make_client(
        activities=[
            SimpleNamespace(
                activity_type="FILL",
                client_order_id="discipline_2026W45",
                symbol="TLT",
            )
        ]
    )

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "ABORTED_ALREADY_FILLED"
    assert result["orders_submitted"] == []
    client.submit_order.assert_not_called()


def test_idempotency_cycle_aborts_on_existing_open_order():
    """An existing open order with this ISO week's tag also prevents duplicates."""
    client = make_client(
        orders=[SimpleNamespace(client_order_id="discipline_2026W45")]
    )

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "ABORTED_ALREADY_FILLED"
    assert result["orders_submitted"] == []
    client.submit_order.assert_not_called()


def test_tuesday_execution_when_monday_is_a_market_holiday():
    """Monday holiday -> market clock closed; Tuesday open executes the same ISO week."""
    monday = datetime(2026, 1, 19, 17, 0, tzinfo=timezone.utc)  # MLK Day, noon ET
    tuesday = datetime(2026, 1, 20, 17, 0, tzinfo=timezone.utc)  # noon ET

    client = make_client()
    client.get_clock.side_effect = [
        SimpleNamespace(is_open=False, timestamp=monday),
        SimpleNamespace(is_open=True, timestamp=tuesday),
    ]

    monday_result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=monday,
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )
    assert monday_result["status"] == "SKIPPED_MARKET_CLOSED"
    assert monday_result["orders_submitted"] == []
    client.submit_order.assert_not_called()

    tuesday_result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=tuesday,
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )
    assert tuesday_result["status"] == "SUCCESS"
    assert tuesday_result["client_order_id"] == "discipline_2026W04"
    client.submit_order.assert_called_once()
    request = client.submit_order.call_args.args[0]
    assert request.client_order_id == "discipline_2026W04"
    assert float(request.notional) == pytest.approx(BASE_DEPOSIT)


def test_cash_cap_uses_account_cash_not_buying_power():
    """Order notional is capped by account.cash, ignoring higher buying_power."""
    client = make_client(cash="50.00", buying_power="20000.00")

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "SUCCESS"
    assert result["deposit"] == pytest.approx(125.0)
    assert result["deployed"] == pytest.approx(50.0)
    assert result["available_cash"] == pytest.approx(50.0)
    client.submit_order.assert_called_once()
    request = client.submit_order.call_args.args[0]
    assert float(request.notional) == pytest.approx(50.0)


def test_cash_cap_respects_lower_non_marginable_buying_power():
    """Available cash is min(account.cash, non_marginable_buying_power)."""
    client = make_client(
        cash="500.00",
        buying_power="20000.00",
        non_marginable_buying_power="40.00",
    )

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "SUCCESS"
    assert result["deployed"] == pytest.approx(40.0)
    request = client.submit_order.call_args.args[0]
    assert float(request.notional) == pytest.approx(40.0)


def test_insufficient_cash_skips_cleanly():
    """available_cash < $1.00 -> clean skip, no order submitted."""
    client = make_client(cash="0.50", buying_power="20000.00")

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "SKIPPED_INSUFFICIENT_CASH"
    assert result["orders_submitted"] == []
    assert "Insufficient non-margin cash" in result["reason"]
    client.submit_order.assert_not_called()


def test_idempotency_prefix_matches_retry_and_sell_suffixes():
    """Retry and sell suffixes share the ISO-week prefix; lookalikes do not."""
    current_dt = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)  # 2026-W45

    for cid in (
        "discipline_2026W45_r1",
        "discipline_2026W45_r2_sell",
        "discipline_2026W45_sell",
    ):
        fills = [SimpleNamespace(activity_type="FILL", client_order_id=cid)]
        assert has_filled_this_week(fills, current_date=current_dt) is True

    lookalike = [
        SimpleNamespace(
            activity_type="FILL",
            client_order_id="discipline_2026W450",
        )
    ]
    assert has_filled_this_week(lookalike, current_date=current_dt) is False


def test_get_existing_orders_requests_all_statuses():
    """Bulk order lookup must include closed orders (rejected/canceled/expired)."""

    class FakeClient:
        def __init__(self):
            self.request = None

        def get_orders(self, request=None):
            self.request = request
            return []

    client = FakeClient()
    assert get_existing_orders(client) == []
    assert client.request is not None
    assert client.request.status.value == "all"


def test_get_existing_orders_supports_no_argument_client():
    """Older/custom clients that do not accept GetOrdersRequest still work."""
    order = SimpleNamespace(
        client_order_id="discipline_2026W45", status="rejected", filled_qty="0"
    )

    class FakeClient:
        def get_orders(self):
            return [order]

    assert get_existing_orders(FakeClient()) == [order]


@pytest.mark.parametrize("status", ["rejected", "canceled", "expired"])
def test_terminal_failure_allows_retry_with_non_colliding_client_order_id(status):
    """A zero-fill rejected/canceled/expired Monday order must not block Tuesday."""
    base = "discipline_2026W45"
    client = make_client(
        orders=[
            SimpleNamespace(
                client_order_id=base,
                status=status,
                filled_qty="0",
            )
        ]
    )

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "SUCCESS"
    assert result["client_order_id"] == f"{base}_r1"
    assert len(result["orders_submitted"]) == 1
    client.submit_order.assert_called_once()
    request = client.submit_order.call_args.args[0]
    assert request.client_order_id == f"{base}_r1"


def test_retry_suffix_advances_after_r1_terminal_failure():
    """If _r1 also failed terminally, the next retry must use _r2."""
    base = "discipline_2026W45"
    client = make_client(
        orders=[
            SimpleNamespace(
                client_order_id=base, status="rejected", filled_qty="0"
            ),
            SimpleNamespace(
                client_order_id=f"{base}_r1", status="canceled", filled_qty="0"
            ),
        ]
    )

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "SUCCESS"
    request = client.submit_order.call_args.args[0]
    assert request.client_order_id == f"{base}_r2"


@pytest.mark.parametrize(
    ("status", "filled_qty"),
    [
        ("filled", "125"),
        ("accepted", "0"),
        ("new", "0"),
        ("partially_filled", "5"),
    ],
)
def test_filled_or_active_order_blocks_duplicate_execution(status, filled_qty):
    """Filled and in-flight orders still abort the week to prevent duplicates."""
    client = make_client(
        orders=[
            SimpleNamespace(
                client_order_id="discipline_2026W45",
                status=status,
                filled_qty=filled_qty,
            )
        ]
    )

    result = run_discipline_cycle(
        trading_client=client,
        paper=True,
        dry_run=False,
        base_deposit=BASE_DEPOSIT,
        current_date=datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc),
        mock_prices=MARKET_PRICES,
        mock_positions=POSITIONS,
    )

    assert result["status"] == "ABORTED_ALREADY_FILLED"
    assert result["orders_submitted"] == []
    client.submit_order.assert_not_called()


def test_terminal_failure_with_partial_fill_still_blocks():
    """A canceled/expired order with any fill is not a clean retry."""
    current_dt = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)
    partial = SimpleNamespace(
        client_order_id="discipline_2026W45",
        status="canceled",
        filled_qty="0.5",
    )
    assert has_existing_week_order([partial], current_date=current_dt) is True
