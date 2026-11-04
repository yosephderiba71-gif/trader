import json
from pathlib import Path

import pandas as pd
import pytest

from market_data import (
    AlpacaMarketDataProvider,
    CsvMarketDataProvider,
    discover_csv_files,
    infer_symbol,
    make_provider,
    normalize_ohlcv,
)


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_normalize_epoch_ms_to_utc():
    frame = pd.DataFrame(
        {
            "timestamp": [1_672_617_600_000, 1_672_617_900_000],
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 100.0],
            "close": [100.5, 101.0],
            "volume": [1000.0, 1500.0],
        }
    )
    out = normalize_ohlcv(frame, "TEST", "unit.csv")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is not None
    assert out.index.name is None
    assert [str(ts) for ts in out.index] == [
        "2023-01-02 00:00:00+00:00",
        "2023-01-02 00:05:00+00:00",
    ]
    assert out.attrs["symbol"] == "TEST"
    assert out.attrs["source"] == "unit.csv"
    assert out.attrs["invalid_ohlc"] == 0


def test_normalize_date_column_without_volume():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.0, 10.0],
            "close": [10.5, 11.0],
        }
    )
    out = normalize_ohlcv(frame, "D", "unit.csv")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["volume"].tolist() == [0.0, 0.0]
    assert out.attrs["invalid_ohlc"] == 0


def test_normalize_counts_invalid_ohlc_without_repair():
    frame = pd.DataFrame(
        {
            "timestamp": [1_672_617_600_000],
            "open": [100.0],
            "high": [99.0],  # invalid: high below open/close
            "low": [98.0],
            "close": [100.0],
            "volume": [1.0],
        }
    )
    out = normalize_ohlcv(frame, "BAD", "bad.csv")
    assert len(out) == 1
    assert out.attrs["invalid_ohlc"] == 1


def test_normalize_empty_frame_still_has_schema():
    out = normalize_ohlcv(pd.DataFrame(), "EMPTY", "empty.csv")
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.attrs["invalid_ohlc"] == 0


def test_infer_symbol_aliases():
    assert infer_symbol("xauusd_5m") == "XAUUSD"
    assert infer_symbol("gold_daily_gc_f_2010") == "GC=F"
    assert infer_symbol("spy_5m_yahoo") == "SPY"
    assert infer_symbol("AAPL_daily") == "AAPL"


def test_discover_csv_files_skips_non_ohlcv(tmp_path):
    _write_csv(
        tmp_path / "good.csv",
        "date,open,high,low,close,volume\n2024-01-02,10,11,9,10.5,100\n",
    )
    _write_csv(tmp_path / "not_bars.csv", "trade_id,pnl\n1,10\n")
    _write_csv(tmp_path / "broken.csv", "open,high,low,close\n,10,11,9\n")

    found = discover_csv_files([tmp_path])
    assert list(found) == ["GOOD"]
    assert found["GOOD"] == tmp_path / "good.csv"


def test_csv_provider_reads_and_filters(tmp_path):
    path = _write_csv(
        tmp_path / "TEST.csv",
        (
            "timestamp,open,high,low,close,volume\n"
            "1672617600000,100,101,99,100.5,1000\n"
            "1672617900000,100.5,101.5,100,101.0,1500\n"
            "1672618200000,101,102,100.5,101.5,1800\n"
        ),
    )
    provider = CsvMarketDataProvider(files={"TEST": path})
    assert provider.list_symbols() == ["TEST"]

    full = provider.get_bars("TEST")
    assert len(full) == 3
    assert full.attrs["source"] == str(path)

    end = pd.Timestamp("2023-01-02T00:05:00Z")
    partial = provider.get_bars("TEST", end=end)
    assert len(partial) == 2
    assert str(partial.index[-1]) == "2023-01-02 00:05:00+00:00"

    start = pd.Timestamp("2023-01-02T00:05:00Z")
    partial = provider.get_bars("TEST", start=start)
    assert len(partial) == 2

    with pytest.raises(KeyError):
        provider.get_bars("MISSING")


def test_csv_provider_default_roots():
    provider = CsvMarketDataProvider()
    assert "XAUUSD" in provider.list_symbols()
    assert "GC=F" in provider.list_symbols()
    assert "SPY" in provider.list_symbols()


def test_make_provider_returns_csv():
    provider = make_provider("csv", roots=("data", "intraday"))
    assert isinstance(provider, CsvMarketDataProvider)
    assert provider.name == "csv"


def test_make_provider_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_provider("not_a_provider")


def test_alpaca_provider_is_data_only():
    # It must not expose trading/order methods or use a TradingClient.
    assert not hasattr(AlpacaMarketDataProvider, "submit_bracket_buy")
    assert not hasattr(AlpacaMarketDataProvider, "close_long")
    assert not hasattr(AlpacaMarketDataProvider, "cancel_order_by_id")

    source = Path(AlpacaMarketDataProvider.__module__ + ".py").resolve()
    text = source.read_text(encoding="utf-8")
    assert "TradingClient" not in text
    assert "submit_order" not in text
    assert "close_position" not in text
    assert "cancel_order" not in text


def test_provider_contract_swap_is_transparent():
    class FakeProvider:
        name = "fake"

        def list_symbols(self):
            return ["FAKE"]

        def get_bars(self, symbol, start=None, end=None, timeframe="1Day"):
            frame = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000.0],
                },
                index=pd.DatetimeIndex(["2024-01-02"], name=None, tz="UTC"),
            )
            frame.attrs.update(symbol=symbol, source="fake", invalid_ohlc=0)
            return frame

    provider = FakeProvider()
    assert provider.name == "fake"
    assert provider.list_symbols() == ["FAKE"]
    assert float(provider.get_bars("FAKE")["close"].iloc[0]) == 100.5
