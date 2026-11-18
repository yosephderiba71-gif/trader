"""Unit tests for the standalone Alpaca paper reconciliation script."""
from __future__ import annotations

import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from dev_tools import reconcile as reconcile_module

EASTERN_TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=EASTERN_TZ)
WEEK_ID = "discipline_2026W47"


class FakeClient:
    def __init__(self, positions=None, activities=None):
        self.positions = positions or []
        self.activities = activities or []

    def get_positions(self):
        return self.positions

    def get_fill_activities(self):
        return self.activities


def clean_positions():
    return [
        {"symbol": "SPY", "qty": "160", "market_value": "80000"},
        {"symbol": "TLT", "qty": "200", "market_value": "20000"},
    ]


def filled_week():
    return [{"activity_type": "FILL", "client_order_id": WEEK_ID}]


def finding_codes(result):
    return [finding.code for finding in result.findings]


@pytest.mark.parametrize(
    "url",
    [
        "https://api.alpaca.markets",
        "http://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.example",
        "https://paper-api.alpaca.markets@evil.example",
        "",
    ],
)
def test_assert_paper_endpoint_rejects_non_paper_urls(url):
    with pytest.raises(ValueError, match="non-paper"):
        reconcile_module.assert_paper_endpoint(url)


def test_assert_paper_endpoint_accepts_paper_url():
    assert (
        reconcile_module.assert_paper_endpoint("https://paper-api.alpaca.markets/")
        == reconcile_module.PAPER_API_BASE_URL
    )


def test_resolve_paper_base_url_rejects_live_env_override():
    with pytest.raises(ValueError, match="non-paper"):
        reconcile_module.resolve_paper_base_url(
            {"ALPACA_BASE_URL": "https://api.alpaca.markets"}
        )

    assert (
        reconcile_module.resolve_paper_base_url(
            {"ALPACA_BASE_URL": "https://paper-api.alpaca.markets"}
        )
        == reconcile_module.PAPER_API_BASE_URL
    )


def test_credentials_missing_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Missing Alpaca"):
        reconcile_module._credentials({})


def test_client_reads_positions_and_fills_only_from_paper_host():
    calls = []

    def http_get(url, headers):
        calls.append((url, dict(headers)))
        path = urllib.parse.urlparse(url).path
        if path == "/v2/positions":
            return clean_positions()
        if path == "/v2/account/activities":
            return filled_week()
        raise AssertionError(f"unexpected path {path}")

    client = reconcile_module.AlpacaPaperClient(
        key="paper-key",
        secret="paper-secret",
        http_get=http_get,
    )

    assert client.get_positions() == clean_positions()
    assert client.get_fill_activities() == filled_week()

    for url, headers in calls:
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.hostname == "paper-api.alpaca.markets"
        assert headers["APCA-API-KEY-ID"] == "paper-key"
        assert headers["APCA-API-SECRET-KEY"] == "paper-secret"
    assert "activity_types=FILL" in calls[1][0]


def test_client_rejects_live_base_url_before_any_request():
    calls = []

    def http_get(url, headers):
        calls.append(url)
        return []

    with pytest.raises(ValueError, match="non-paper"):
        reconcile_module.AlpacaPaperClient(
            key="k",
            secret="s",
            base_url="https://api.alpaca.markets",
            http_get=http_get,
        )
    assert calls == []


def test_reconcile_clean_at_target_weights_with_week_fill():
    result = reconcile_module.reconcile(
        FakeClient(clean_positions(), filled_week()),
        now=NOW,
    )

    assert result.ok is True
    assert result.findings == ()
    assert result.week_id == WEEK_ID
    assert result.spy_weight == pytest.approx(0.80)
    assert result.tlt_weight == pytest.approx(0.20)
    assert result.fill_client_order_ids == (WEEK_ID,)


def test_reconcile_flags_weight_drift_beyond_three_percent():
    positions = [
        {"symbol": "SPY", "qty": "180", "market_value": "90000"},
        {"symbol": "TLT", "qty": "100", "market_value": "10000"},
    ]

    result = reconcile_module.reconcile(
        FakeClient(positions, filled_week()),
        now=NOW,
    )

    assert result.ok is False
    assert finding_codes(result).count("WEIGHT_DRIFT") == 2


def test_reconcile_does_not_flag_drift_at_exactly_three_percent():
    positions = [
        {"symbol": "SPY", "qty": "166", "market_value": "83000"},
        {"symbol": "TLT", "qty": "170", "market_value": "17000"},
    ]

    result = reconcile_module.reconcile(
        FakeClient(positions, filled_week()),
        now=NOW,
    )

    assert result.ok is True
    assert result.spy_weight == pytest.approx(0.83)
    assert result.tlt_weight == pytest.approx(0.17)


def test_reconcile_flags_unexpected_symbol():
    positions = [
        *clean_positions(),
        {"symbol": "QQQ", "qty": "10", "market_value": "5000"},
    ]

    result = reconcile_module.reconcile(
        FakeClient(positions, filled_week()),
        now=NOW,
    )

    assert "UNAUTHORIZED_SYMBOL" in finding_codes(result)


def test_reconcile_ignores_zero_quantity_symbols():
    positions = [
        *clean_positions(),
        {"symbol": "QQQ", "qty": "0", "market_value": "0"},
    ]

    result = reconcile_module.reconcile(
        FakeClient(positions, filled_week()),
        now=NOW,
    )

    assert "UNAUTHORIZED_SYMBOL" not in finding_codes(result)


def test_reconcile_flags_missing_week_fill():
    result = reconcile_module.reconcile(
        FakeClient(clean_positions(), []),
        now=NOW,
    )

    assert result.ok is False
    assert finding_codes(result) == ["MISSING_WEEK_FILL"]
    assert result.fill_client_order_ids == ()


def test_reconcile_matches_retry_and_sell_suffixes_for_week():
    activities = [
        {"activity_type": "FILL", "client_order_id": "discipline_2026W46"},
        {"activity_type": "FILL", "client_order_id": f"{WEEK_ID}_r1"},
        {"activity_type": "FILL", "client_order_id": f"{WEEK_ID}_sell"},
    ]

    result = reconcile_module.reconcile(
        FakeClient(clean_positions(), activities),
        now=NOW,
    )

    assert result.ok is True
    assert result.fill_client_order_ids == (
        f"{WEEK_ID}_r1",
        f"{WEEK_ID}_sell",
    )


def test_main_returns_zero_when_clean(monkeypatch, capsys):
    monkeypatch.setattr(
        reconcile_module, "AlpacaPaperClient", lambda: FakeClient(
            clean_positions(), filled_week()
        )
    )
    monkeypatch.setattr(
        reconcile_module, "current_week_id", lambda now=None: WEEK_ID
    )

    assert reconcile_module.main([]) == 0
    output = capsys.readouterr().out
    assert "OK: reconciliation clean." in output
    assert WEEK_ID in output


def test_main_returns_one_on_anomaly(monkeypatch, capsys):
    positions = [
        *clean_positions(),
        {"symbol": "QQQ", "qty": "10", "market_value": "5000"},
    ]
    monkeypatch.setattr(
        reconcile_module, "AlpacaPaperClient", lambda: FakeClient(
            positions, filled_week()
        )
    )
    monkeypatch.setattr(
        reconcile_module, "current_week_id", lambda now=None: WEEK_ID
    )

    assert reconcile_module.main([]) == 1
    output = capsys.readouterr().out
    assert "ANOMALY [UNAUTHORIZED_SYMBOL]" in output


def test_main_returns_one_on_credential_error(monkeypatch, capsys):
    def boom():
        raise RuntimeError("Missing Alpaca API credentials.")

    monkeypatch.setattr(reconcile_module, "AlpacaPaperClient", boom)

    assert reconcile_module.main([]) == 1
    assert "Error: Missing Alpaca API credentials." in capsys.readouterr().err
