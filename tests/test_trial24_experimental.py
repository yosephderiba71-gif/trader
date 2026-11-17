"""Tests for the isolated Trial #24 experimental runner."""
from __future__ import annotations

import pytest

from momentum_bot import trial24_experimental as trial24


def test_build_trial24_clients_uses_only_dedicated_env(monkeypatch):
    captured = {}

    class FakeTradingClient:
        def __init__(self, api_key, api_secret, paper):
            captured["trading"] = (api_key, api_secret, paper)

    class FakeDataClient:
        def __init__(self, api_key, api_secret):
            captured["data"] = (api_key, api_secret)

    monkeypatch.setenv("TRIAL24_ALPACA_API_KEY", "fake-trial-key")
    monkeypatch.setenv("TRIAL24_ALPACA_SECRET_KEY", "fake-trial-secret")
    monkeypatch.setenv("ALPACA_API_KEY", "main-account-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "main-account-secret")
    monkeypatch.setattr("alpaca.trading.client.TradingClient", FakeTradingClient)
    monkeypatch.setattr(
        "alpaca.data.historical.StockHistoricalDataClient", FakeDataClient
    )

    trading, data = trial24.build_trial24_clients(paper=True)

    assert isinstance(trading, FakeTradingClient)
    assert isinstance(data, FakeDataClient)
    assert captured["trading"] == ("fake-trial-key", "fake-trial-secret", True)
    assert captured["data"] == ("fake-trial-key", "fake-trial-secret")


def test_missing_trial24_credentials_do_not_fall_back(monkeypatch):
    monkeypatch.delenv("TRIAL24_ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("TRIAL24_ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "main-account-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "main-account-secret")

    with pytest.raises(RuntimeError, match="TRIAL24_ALPACA_API_KEY"):
        trial24.build_trial24_clients(paper=True)


def test_build_trial24_clients_rejects_non_paper():
    with pytest.raises(ValueError, match="paper=True"):
        trial24.build_trial24_clients(paper=False)


def test_main_passes_dedicated_clients_to_rebalance(monkeypatch):
    trading = object()
    data = object()
    captured = {}

    def fake_build_clients(paper=True):
        captured["paper"] = paper
        return trading, data

    def fake_rebalance(targets, **kwargs):
        captured["targets"] = targets
        captured.update(kwargs)
        return []

    monkeypatch.setattr(trial24, "build_trial24_clients", fake_build_clients)
    monkeypatch.setattr(trial24, "fetch_prices", lambda period="2y": object())
    monkeypatch.setattr(trial24, "compute_metrics", lambda prices: [])
    monkeypatch.setattr(trial24, "compute_target_weights", lambda prices: {"SPY": 1.0})
    monkeypatch.setattr(trial24, "latest_prices", lambda prices: {"SPY": 500.0})
    monkeypatch.setattr(trial24, "rebalance", fake_rebalance)

    assert trial24.main([]) == 0
    assert captured["paper"] is True
    assert captured["dry_run"] is True
    assert captured["trading_client"] is trading
    assert captured["data_client"] is data
    assert captured["require_first_trading_day"] is True
