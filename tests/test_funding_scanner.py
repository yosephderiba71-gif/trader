"""Unit tests for the read-only crypto funding-rate and basis scanner."""
from __future__ import annotations

import sqlite3
import stat
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import pytest

from market_tools import funding_scanner as scanner

STAMP = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


def _symbol(url: str) -> str:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get(
        "symbol", ["BTCUSDT"]
    )[0]


def fake_http(url: str):
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    symbol = params.get("symbol", ["BTCUSDT"])[0]

    if parsed.hostname == "fapi.binance.com":
        assert parsed.path == "/fapi/v1/premiumIndex"
        return {
            "symbol": symbol,
            "markPrice": "50100.0",
            "lastFundingRate": "0.0001",
        }
    if parsed.hostname == "api.binance.com":
        assert parsed.path == "/api/v3/ticker/price"
        return {"symbol": symbol, "price": "50000.0"}
    if parsed.hostname == "api.bybit.com":
        assert parsed.path == "/v5/market/tickers"
        category = params.get("category", [""])[0]
        if category == "linear":
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "symbol": symbol,
                            "markPrice": "50150.0",
                            "fundingRate": "0.0002",
                        }
                    ]
                },
            }
        if category == "spot":
            return {
                "retCode": 0,
                "result": {"list": [{"symbol": symbol, "lastPrice": "50000.0"}]},
            }
    raise AssertionError(f"unexpected URL: {url}")


def test_annualized_funding_apr_and_basis_math():
    snapshot = scanner.fetch_binance_snapshot(
        "BTCUSDT",
        http_get=fake_http,
        timestamp_utc=STAMP,
    )

    assert snapshot.status == scanner.STATUS_OK
    assert snapshot.funding_rate == pytest.approx(0.0001)
    assert snapshot.funding_interval_hours == 8.0
    assert snapshot.annualized_funding_apr_pct == pytest.approx(10.95)
    assert snapshot.spot_price == pytest.approx(50000.0)
    assert snapshot.perp_price == pytest.approx(50100.0)
    assert snapshot.basis_pct == pytest.approx(0.2)

    assert scanner.annualized_funding_apr_pct(0.0001) == pytest.approx(10.95)
    assert scanner.annualized_funding_apr_pct(0.0001, 4.0) == pytest.approx(21.9)
    assert scanner.basis_pct(50100.0, 50000.0) == pytest.approx(0.2)
    assert scanner.basis_pct(49900.0, 50000.0) == pytest.approx(-0.2)


def test_bybit_snapshot_parses_linear_and_spot_tickers():
    snapshot = scanner.fetch_bybit_snapshot(
        "BTCUSDT",
        http_get=fake_http,
        timestamp_utc=STAMP,
    )

    assert snapshot.status == scanner.STATUS_OK
    assert snapshot.funding_rate == pytest.approx(0.0002)
    assert snapshot.annualized_funding_apr_pct == pytest.approx(21.9)
    assert snapshot.perp_price == pytest.approx(50150.0)
    assert snapshot.spot_price == pytest.approx(50000.0)
    assert snapshot.basis_pct == pytest.approx(0.3)


def test_cross_venue_funding_spread_apr_pct():
    binance = scanner.fetch_binance_snapshot("BTCUSDT", http_get=fake_http)
    bybit = scanner.fetch_bybit_snapshot("BTCUSDT", http_get=fake_http)

    spread = scanner.cross_venue_funding_spread_apr_pct(binance, bybit)

    assert spread == pytest.approx(10.95 - 21.9)
    assert scanner.cross_venue_funding_spread_apr_pct(binance, None) is None

    failed = scanner.FundingSnapshot(
        timestamp_utc=binance.timestamp_utc,
        exchange="bybit",
        symbol="BTCUSDT",
        funding_rate=None,
        funding_interval_hours=8.0,
        annualized_funding_apr_pct=None,
        spot_price=None,
        perp_price=None,
        basis_pct=None,
        status=scanner.STATUS_FETCH_ERROR,
        error_message="HTTPError: 451",
    )
    assert scanner.cross_venue_funding_spread_apr_pct(binance, failed) is None

    other_symbol = scanner.fetch_binance_snapshot("ETHUSDT", http_get=fake_http)
    with pytest.raises(ValueError, match="same symbol"):
        scanner.cross_venue_funding_spread_apr_pct(other_symbol, bybit)


def test_sqlite_wal_permissions_and_row_insertion(tmp_path):
    db_path = tmp_path / "funding_rates.db"
    snapshots = scanner.collect_snapshots(
        symbols=["BTCUSDT"],
        http_get=fake_http,
        timestamp_utc=STAMP,
    )

    assert scanner.log_snapshots(snapshots, db_path) == 2

    with sqlite3.connect(db_path) as raw_connection:
        journal_mode = raw_connection.execute("PRAGMA journal_mode").fetchone()[0]
        table_columns = [
            row[1]
            for row in raw_connection.execute(
                "PRAGMA table_info(funding_rates)"
            ).fetchall()
        ]
        rows = raw_connection.execute(
            "SELECT exchange, symbol, status FROM funding_rates ORDER BY id"
        ).fetchall()

    assert journal_mode.lower() == "wal"
    assert table_columns == scanner.TABLE_COLUMNS
    assert rows == [("binance", "BTCUSDT", "OK"), ("bybit", "BTCUSDT", "OK")]
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600

    connection = scanner.connect_funding_db(db_path)
    try:
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        connection.close()
    assert busy_timeout == scanner.SQLITE_BUSY_TIMEOUT_MS == 5000

    logged = scanner.load_funding_rates(db_path)
    assert len(logged) == 2
    binance_row = logged[0]
    assert binance_row["exchange"] == "binance"
    assert binance_row["timestamp_utc"] == STAMP.isoformat(timespec="seconds")
    assert binance_row["funding_rate"] == pytest.approx(0.0001)
    assert binance_row["annualized_funding_apr_pct"] == pytest.approx(10.95)
    assert binance_row["spot_price"] == pytest.approx(50000.0)
    assert binance_row["perp_price"] == pytest.approx(50100.0)
    assert binance_row["basis_pct"] == pytest.approx(0.2)
    assert binance_row["error_message"] == ""


def test_fetch_error_is_logged_when_one_exchange_fails(tmp_path):
    def geo_blocked(url: str):
        if urllib.parse.urlparse(url).hostname == "api.bybit.com":
            raise urllib.error.HTTPError(
                url,
                451,
                "Unavailable For Legal Reasons",
                {},
                None,
            )
        return fake_http(url)

    snapshots = scanner.collect_snapshots(
        symbols=["BTCUSDT"],
        http_get=geo_blocked,
        timestamp_utc=STAMP,
    )

    by_exchange = {snapshot.exchange: snapshot for snapshot in snapshots}
    assert by_exchange["binance"].status == scanner.STATUS_OK
    assert by_exchange["bybit"].status == scanner.STATUS_FETCH_ERROR
    assert "451" in by_exchange["bybit"].error_message
    assert by_exchange["bybit"].funding_rate is None

    db_path = tmp_path / "funding_rates.db"
    assert scanner.log_snapshots(snapshots, db_path) == 2

    rows = scanner.load_funding_rates(db_path)
    bybit_row = next(row for row in rows if row["exchange"] == "bybit")
    assert bybit_row["status"] == scanner.STATUS_FETCH_ERROR
    assert "451" in bybit_row["error_message"]
    assert bybit_row["annualized_funding_apr_pct"] is None


def test_one_failed_pair_does_not_stop_remaining_pairs():
    def bybit_btc_blocked(url: str):
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname == "api.bybit.com" and _symbol(url) == "BTCUSDT":
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        return fake_http(url)

    snapshots = scanner.collect_snapshots(
        symbols=["BTCUSDT", "ETHUSDT"],
        http_get=bybit_btc_blocked,
        timestamp_utc=STAMP,
    )

    assert len(snapshots) == 4
    statuses = {
        (snapshot.symbol, snapshot.exchange): snapshot.status
        for snapshot in snapshots
    }
    assert statuses[("BTCUSDT", "binance")] == scanner.STATUS_OK
    assert statuses[("BTCUSDT", "bybit")] == scanner.STATUS_FETCH_ERROR
    assert statuses[("ETHUSDT", "binance")] == scanner.STATUS_OK
    assert statuses[("ETHUSDT", "bybit")] == scanner.STATUS_OK


def test_invalid_payload_logs_fetch_error_instead_of_crashing():
    def bad_bybit(url: str):
        if "api.bybit.com" in url:
            return {"retCode": 10001, "retMsg": "bad symbol", "result": {}}
        return fake_http(url)

    snapshot = scanner.fetch_bybit_snapshot("BTCUSDT", http_get=bad_bybit)

    assert snapshot.status == scanner.STATUS_FETCH_ERROR
    assert "retCode=10001" in snapshot.error_message


def test_http_get_uses_ten_second_timeout(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(scanner.urllib.request, "urlopen", fake_urlopen)

    assert scanner._http_get_json("https://example.test") == {"ok": True}
    assert captured["timeout"] == 10.0


def test_main_logs_rows_and_prints_cross_venue_spread(tmp_path, capsys):
    db_path = tmp_path / "funding_rates.db"

    exit_code = scanner.main(
        ["--db-path", str(db_path), "--symbols", "BTCUSDT"],
        http_get=fake_http,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Cross-venue funding APR spread (Binance - Bybit):" in output
    assert "BTCUSDT: -10.95 pp" in output
    assert f"Logged 2 row(s) to {db_path}" in output
    assert len(scanner.load_funding_rates(db_path)) == 2


def test_main_returns_one_when_every_fetch_fails(tmp_path):
    def always_fails(url: str):
        raise urllib.error.URLError("network down")

    exit_code = scanner.main(
        ["--db-path", str(tmp_path / "funding_rates.db"), "--symbols", "BTCUSDT"],
        http_get=always_fails,
    )

    assert exit_code == 1
    rows = scanner.load_funding_rates(tmp_path / "funding_rates.db")
    assert len(rows) == 2
    assert all(row["status"] == scanner.STATUS_FETCH_ERROR for row in rows)
