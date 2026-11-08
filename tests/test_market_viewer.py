import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from market_data import CsvMarketDataProvider
from market_viewer import build_payload, default_provider, make_server, render_html, write_static


def _provider(tmp_path: Path) -> CsvMarketDataProvider:
    bars_one = (
        "date,open,high,low,close,volume\n"
        "2024-01-02,100,101,99,100.5,1000\n"
        "2024-01-03,100.5,101.5,100,101.0,1500\n"
    )
    bars_two = (
        "timestamp,open,high,low,close\n"
        "1704067200000,200,202,199,201.0\n"
    )
    (tmp_path / "GOOD.csv").write_text(bars_one, encoding="utf-8")
    (tmp_path / "OTHER.csv").write_text(bars_two, encoding="utf-8")
    (tmp_path / "trades.csv").write_text("trade_id,pnl\n1,10\n", encoding="utf-8")
    return CsvMarketDataProvider(roots=[tmp_path])


def test_default_provider_finds_local_assets():
    provider = default_provider()
    symbols = provider.list_symbols()
    assert "SPY" in symbols


def test_build_payload_multi_asset(tmp_path):
    payload = build_payload(_provider(tmp_path), max_bars=50)
    assert payload["mode"] == "read-only"
    assert payload["provider"] == "csv"
    assert [asset["symbol"] for asset in payload["assets"]] == ["GOOD", "OTHER"]
    assert payload["data_health"]["assets"] == 2
    assert payload["data_health"]["errors"] == []

    good = payload["assets"][0]
    assert good["last_close"] == pytest.approx(101.0)
    assert good["change"] == pytest.approx(0.5)
    assert len(good["bars"]) == 2
    assert good["invalid_ohlc"] == 0

    other = payload["assets"][1]
    assert other["last_close"] == pytest.approx(201.0)
    assert other["period_volume"] == 0.0


def test_build_payload_read_only_security_fields(tmp_path):
    payload = build_payload(_provider(tmp_path), max_bars=50)
    security = payload["security"]
    assert security["read_only"] is True
    assert security["api_keys"] is False
    assert security["broker_orders"] is False
    assert security["external_network"] is False
    assert security["writes"] is False
    assert security["bind"] == "127.0.0.1"


def test_build_payload_excludes_journal_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "fake-secret")
    payload = build_payload(_provider(tmp_path), max_bars=50)
    text = json.dumps(payload, default=str).lower()
    assert "fake-key" not in text
    assert "fake-secret" not in text
    assert "alpaca_api_key" not in text
    assert "alpaca_api_secret" not in text


def test_render_html_is_local_only_and_read_only(tmp_path):
    payload = build_payload(_provider(tmp_path), max_bars=50)
    html = render_html(payload)
    assert "<script src=" not in html
    assert "http://" not in html
    assert "https://" not in html
    assert "submit_order" not in html
    assert "TradingClient" not in html
    assert "127.0.0.1" in html
    assert "LOCALHOST ONLY" in html


def test_render_html_escapes_symbols(tmp_path):
    provider = _provider(tmp_path)
    payload = build_payload(provider, max_bars=50)
    payload["assets"][0]["symbol"] = '<script>alert("x")</script>'
    html = render_html(payload)
    assert "<script>alert(\"x\")</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_html_can_refresh(tmp_path):
    payload = build_payload(_provider(tmp_path), max_bars=50)
    assert "http-equiv=\"refresh\"" not in render_html(payload)
    assert "http-equiv=\"refresh\"" in render_html(payload, refresh_seconds=60)


def test_write_static(tmp_path):
    payload = build_payload(_provider(tmp_path), max_bars=50)
    target = write_static(payload, tmp_path / "viewer.html", refresh_seconds=60)
    assert target.exists()
    html = target.read_text(encoding="utf-8")
    assert "Read-only multi-asset viewer" in html
    assert "Market Viewer" in html


def test_server_routes_are_read_only(tmp_path):
    payload_factory = lambda: build_payload(_provider(tmp_path), max_bars=50)  # noqa: E731
    server = make_server(payload_factory, port=0)
    thread = None
    try:
        host, port = server.server_address[:2]
        base = f"http://{host}:{port}"
        import threading

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        health = urllib.request.urlopen(f"{base}/health", timeout=5)
        assert health.status == 200

        data = urllib.request.urlopen(f"{base}/api/data", timeout=5)
        assert data.status == 200
        payload = json.loads(data.read().decode())
        assert payload["mode"] == "read-only"

        page = urllib.request.urlopen(f"{base}/", timeout=5)
        assert page.status == 200
        assert b"Market Viewer" in page.read()

        for method in ("POST", "PUT", "DELETE"):
            request = urllib.request.Request(base + "/", method=method)
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(request, timeout=5)
            assert exc_info.value.code == 405

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/missing", timeout=5)
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_payload_does_not_include_local_paths(tmp_path):
    payload = build_payload(_provider(tmp_path), max_bars=50)
    text = json.dumps(payload, default=str)
    assert "/Users/" not in text
    assert "ALPACA_API" not in text
