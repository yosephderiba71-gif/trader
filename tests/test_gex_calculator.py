"""Unit tests for the read-only SPY GEX calculator and snapshot logger."""
from __future__ import annotations

import math
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import market_tools.gex_calculator as gex_calculator
from market_tools.gex_calculator import (
    HISTORY_COLUMNS,
    STATUS_FETCH_ERROR,
    STATUS_OK,
    GexInputs,
    GexSnapshot,
    aggregate_net_gex_at_spot,
    black_scholes_gamma,
    calculate_gex_snapshot,
    calculate_put_call_oi_ratio,
    calculate_strike_gex,
    collect_gex_snapshot,
    fetch_gex_inputs,
    find_call_wall,
    find_gamma_flip,
    find_max_pain,
    find_put_wall,
    gamma_flip_distance,
    gamma_regime,
    load_gex_history,
    log_gex_snapshot,
    norm_pdf,
    select_nearest_expiration,
    time_to_expiry_years,
)

NOW_ET = datetime(2026, 11, 17, 12, 0, tzinfo=ZoneInfo("America/New_York"))
EXPIRATION = "2026-11-20"


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["strike", "openInterest", "impliedVolatility"])


def _calls() -> pd.DataFrame:
    return _frame(
        [
            {"strike": 500.0, "openInterest": 100, "impliedVolatility": 0.20},
            {"strike": 510.0, "openInterest": 250, "impliedVolatility": 0.22},
            {"strike": 520.0, "openInterest": 80, "impliedVolatility": 0.24},
        ]
    )


def _puts() -> pd.DataFrame:
    return _frame(
        [
            {"strike": 480.0, "openInterest": 300, "impliedVolatility": 0.24},
            {"strike": 490.0, "openInterest": 120, "impliedVolatility": 0.22},
            {"strike": 500.0, "openInterest": 90, "impliedVolatility": 0.21},
        ]
    )


class FakeTicker:
    def __init__(self, spot=500.0, calls=None, puts=None, expirations=(EXPIRATION,)):
        self.fast_info = {"last_price": spot}
        self.options = expirations
        self._calls = calls
        self._puts = puts

    def option_chain(self, expiration):
        return SimpleNamespace(
            calls=self._calls if self._calls is not None else pd.DataFrame(),
            puts=self._puts if self._puts is not None else pd.DataFrame(),
        )


def _snapshot(
    date_et: str,
    *,
    status: str = STATUS_OK,
    error_message: str = "",
) -> GexSnapshot:
    ok = status == STATUS_OK
    return GexSnapshot(
        symbol="SPY",
        date_et=date_et,
        timestamp_et=f"{date_et}T12:00:00-05:00",
        expiration=EXPIRATION if ok else None,
        spot=500.0 if ok else None,
        call_wall=510.0 if ok else None,
        put_wall=480.0 if ok else None,
        gamma_flip=505.0 if ok else None,
        net_gex_total=1_250_000_000.0 if ok else None,
        status=status,
        error_message=error_message,
    )


def test_norm_pdf_known_values():
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
    assert norm_pdf(1.0) == pytest.approx(math.exp(-0.5) / math.sqrt(2.0 * math.pi))


def test_black_scholes_gamma_matches_formula():
    spot = 100.0
    strike = 100.0
    time_to_expiry = 30.0 / 365.0
    iv = 0.20
    rate = 0.045

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv**2) * time_to_expiry) / (iv * sqrt_t)
    expected = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) / (spot * iv * sqrt_t)

    assert black_scholes_gamma(spot, strike, time_to_expiry, iv, rate) == pytest.approx(expected)


def test_zero_dte_gamma_floor_prevents_division_by_zero():
    floor = time_to_expiry_years(0)
    assert floor == pytest.approx(1.0 / (365.0 * 24.0))
    assert floor > 0.0

    gamma = black_scholes_gamma(500.0, 500.0, floor, 0.20)
    assert math.isfinite(gamma)
    assert gamma > 0.0
    assert time_to_expiry_years(1) == pytest.approx(1.0 / 365.0)


def test_invalid_option_rows_are_filtered_and_walls_identified():
    calls = _frame(
        [
            {"strike": 100.0, "openInterest": 10, "impliedVolatility": 0.20},
            {"strike": 105.0, "openInterest": 50, "impliedVolatility": 0.20},
            {"strike": 110.0, "openInterest": float("nan"), "impliedVolatility": 0.20},
            {"strike": 115.0, "openInterest": 0, "impliedVolatility": 0.20},
        ]
    )
    puts = _frame(
        [
            {"strike": 95.0, "openInterest": 70, "impliedVolatility": 0.21},
            {"strike": 100.0, "openInterest": 30, "impliedVolatility": 0.21},
            {"strike": 105.0, "openInterest": 20, "impliedVolatility": float("nan")},
            {"strike": 110.0, "openInterest": 20, "impliedVolatility": 0.0},
        ]
    )

    strikes = calculate_strike_gex(100.0, calls, puts, days_until_expiry=30)

    assert {row.strike for row in strikes} == {95.0, 100.0, 105.0}
    assert find_call_wall(strikes) == 105.0
    assert find_put_wall(strikes) == 95.0
    assert all(math.isfinite(row.net_gex) for row in strikes)


def test_gamma_flip_interpolates_negative_to_positive_crossing():
    calls = _frame([{"strike": 110.0, "openInterest": 100, "impliedVolatility": 0.20}])
    puts = _frame([{"strike": 90.0, "openInterest": 100, "impliedVolatility": 0.20}])
    strikes = calculate_strike_gex(100.0, calls, puts, days_until_expiry=30)
    time_to_expiry = time_to_expiry_years(30)

    flip = find_gamma_flip(strikes, spot=100.0, time_to_expiry=time_to_expiry)

    assert flip is not None
    assert 90.0 < flip < 110.0
    assert abs(aggregate_net_gex_at_spot(strikes, flip, time_to_expiry)) < 1.0
    assert abs(aggregate_net_gex_at_spot(strikes, flip, time_to_expiry)) < abs(
        aggregate_net_gex_at_spot(strikes, 90.0, time_to_expiry)
    )


def test_gamma_flip_returns_none_without_negative_to_positive_crossing():
    calls = _frame([{"strike": 100.0, "openInterest": 100, "impliedVolatility": 0.20}])
    puts = _frame([])
    strikes = calculate_strike_gex(100.0, calls, puts, days_until_expiry=30)

    assert find_gamma_flip(strikes, spot=100.0, time_to_expiry=time_to_expiry_years(30)) is None


def test_calculate_snapshot_totals_and_regime():
    snapshot = calculate_gex_snapshot(
        spot=500.0,
        expiration=EXPIRATION,
        calls=_calls(),
        puts=_puts(),
        now_et=NOW_ET,
    )

    assert snapshot.status == STATUS_OK
    assert snapshot.spot == pytest.approx(500.0)
    assert snapshot.call_wall == 510.0
    assert snapshot.put_wall == 480.0
    assert snapshot.net_gex_total == pytest.approx(sum(row.net_gex for row in snapshot.strikes))
    assert gamma_regime(snapshot.net_gex_total).startswith("Long Gamma") or gamma_regime(
        snapshot.net_gex_total
    ).startswith("Short Gamma")


def test_select_nearest_non_expired_expiration():
    expirations = ["2026-11-13", "2026-11-20", "2026-11-27"]
    assert select_nearest_expiration(expirations, now_et=NOW_ET) == "2026-11-20"


def test_same_day_csv_deduplication(tmp_path):
    path = tmp_path / "gex_history.csv"

    assert log_gex_snapshot(_snapshot("2026-11-17"), path) == "appended"
    assert log_gex_snapshot(_snapshot("2026-11-17"), path) == "skipped"

    rows = load_gex_history(path)
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-11-17"
    assert rows[0]["status"] == STATUS_OK
    assert list(rows[0].keys()) == HISTORY_COLUMNS


def test_fetch_error_row_is_explicit_and_can_be_replaced_by_success(tmp_path):
    path = tmp_path / "gex_history.csv"

    def failing_factory(symbol):
        raise RuntimeError("network down")

    failed = collect_gex_snapshot(
        history_path=path,
        ticker_factory=failing_factory,
        now_et=NOW_ET,
    )

    assert failed.status == STATUS_FETCH_ERROR
    assert "network down" in failed.error_message
    rows = load_gex_history(path)
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_FETCH_ERROR
    assert "network down" in rows[0]["error_message"]
    assert rows[0]["spot"] == ""

    ticker = FakeTicker(spot=500.0, calls=_calls(), puts=_puts())
    succeeded = collect_gex_snapshot(
        history_path=path,
        ticker_factory=lambda symbol: ticker,
        now_et=NOW_ET,
    )

    assert succeeded.status == STATUS_OK
    rows = load_gex_history(path)
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_OK
    assert rows[0]["spot"] != ""
    assert rows[0]["error_message"] == ""


def test_empty_option_chain_writes_fetch_error_row(tmp_path):
    path = tmp_path / "gex_history.csv"
    ticker = FakeTicker(spot=500.0, calls=None, puts=None)

    snapshot = collect_gex_snapshot(
        history_path=path,
        ticker_factory=lambda symbol: ticker,
        now_et=NOW_ET,
    )

    assert snapshot.status == STATUS_FETCH_ERROR
    assert "empty option chain" in snapshot.error_message
    rows = load_gex_history(path)
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_FETCH_ERROR


def test_repeated_fetch_errors_do_not_duplicate_the_date(tmp_path):
    path = tmp_path / "gex_history.csv"

    def failing_factory(symbol):
        raise TimeoutError("timed out")

    first = collect_gex_snapshot(history_path=path, ticker_factory=failing_factory, now_et=NOW_ET)
    second = collect_gex_snapshot(history_path=path, ticker_factory=failing_factory, now_et=NOW_ET)

    assert first.status == STATUS_FETCH_ERROR
    assert second.status == STATUS_FETCH_ERROR
    rows = load_gex_history(path)
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_FETCH_ERROR
    assert "timed out" in rows[0]["error_message"]


def test_cli_main_collects_and_logs_snapshot(tmp_path, monkeypatch, capsys):
    """The headless CLI entrypoint collects and writes the daily CSV row."""

    def fake_fetch_gex_inputs(symbol, ticker_factory=None, now_et=None):
        return GexInputs(
            symbol=symbol,
            spot=500.0,
            expiration=EXPIRATION,
            calls=_calls(),
            puts=_puts(),
        )

    monkeypatch.setattr(gex_calculator, "fetch_gex_inputs", fake_fetch_gex_inputs)
    path = tmp_path / "gex_history.csv"

    assert gex_calculator.main(["--history-path", str(path)]) == 0

    rows = load_gex_history(path)
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_OK
    assert rows[0]["spot"] != ""
    assert "OK" in capsys.readouterr().out


def test_snapshot_includes_institutional_metrics():
    snapshot = calculate_gex_snapshot(
        spot=500.0,
        expiration=EXPIRATION,
        calls=_calls(),
        puts=_puts(),
        now_et=NOW_ET,
    )

    assert snapshot.put_call_oi_ratio == pytest.approx((300 + 120 + 90) / (100 + 250 + 80))
    assert snapshot.max_pain == 480.0
    if snapshot.gamma_flip is None:
        assert snapshot.dealer_gamma_flip_dist is None
    else:
        assert snapshot.dealer_gamma_flip_dist == pytest.approx(
            abs(500.0 - snapshot.gamma_flip)
        )
    assert snapshot.zero_dte_net_gex is None


def test_zero_dte_net_gex_uses_chain_expiring_today():
    today = NOW_ET.date().isoformat()

    snapshot = calculate_gex_snapshot(
        spot=500.0,
        expiration=today,
        calls=_calls(),
        puts=_puts(),
        now_et=NOW_ET,
    )

    assert snapshot.zero_dte_net_gex is not None
    assert snapshot.zero_dte_net_gex == pytest.approx(snapshot.net_gex_total)


def test_metric_helpers_handle_missing_side():
    calls = _frame([])
    puts = _frame([{"strike": 95.0, "openInterest": 70, "impliedVolatility": 0.21}])
    strikes = calculate_strike_gex(100.0, calls, puts, days_until_expiry=30)

    assert calculate_put_call_oi_ratio(strikes) is None
    assert find_max_pain(strikes) == 95.0
    assert gamma_flip_distance(100.0, None) is None
    assert gamma_flip_distance(None, 100.0) is None
    assert gamma_flip_distance(100.0, 103.5) == pytest.approx(3.5)


def test_history_csv_logs_new_institutional_columns(tmp_path):
    path = tmp_path / "gex_history.csv"
    snapshot = calculate_gex_snapshot(
        spot=500.0,
        expiration=EXPIRATION,
        calls=_calls(),
        puts=_puts(),
        now_et=NOW_ET,
    )

    assert log_gex_snapshot(snapshot, path) == "appended"
    row = load_gex_history(path)[0]

    assert list(row.keys()) == HISTORY_COLUMNS
    for column in (
        "zero_dte_net_gex",
        "put_call_oi_ratio",
        "max_pain",
        "dealer_gamma_flip_dist",
    ):
        assert column in row
    assert float(row["put_call_oi_ratio"]) == pytest.approx((300 + 120 + 90) / (100 + 250 + 80))
    assert float(row["max_pain"]) == 480.0


def test_fetch_gex_inputs_captures_same_day_chain():
    ticker = FakeTicker(
        spot=500.0,
        calls=_calls(),
        puts=_puts(),
        expirations=(NOW_ET.date().isoformat(), EXPIRATION),
    )

    inputs = fetch_gex_inputs(
        ticker_factory=lambda symbol: ticker,
        now_et=NOW_ET,
    )

    assert inputs.expiration == NOW_ET.date().isoformat()
    assert inputs.zero_dte_calls is not None
    assert inputs.zero_dte_puts is not None


def test_fetch_gex_inputs_has_no_zero_dte_chain_on_non_expiry_day():
    ticker = FakeTicker(spot=500.0, calls=_calls(), puts=_puts())

    inputs = fetch_gex_inputs(
        ticker_factory=lambda symbol: ticker,
        now_et=NOW_ET,
    )

    assert inputs.expiration == EXPIRATION
    assert inputs.zero_dte_calls is None
    assert inputs.zero_dte_puts is None
