import numpy as np
import pandas as pd


def make_bars(n: int = 400, seed: int = 0, drift: float = 0.0005, vol: float = 0.01, start: float = 100.0) -> pd.DataFrame:
    """Random-walk hourly bars with valid OHLC relationships."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[start, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.integers(10_000, 50_000, n).astype(float)
    index = pd.date_range("2025-01-02 14:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def flat_bars(n: int, price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2025-01-02 14:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.full(n, price),
            "high": np.full(n, price + 0.5),
            "low": np.full(n, price - 0.5),
            "close": np.full(n, price),
            "volume": np.full(n, 10_000.0),
        },
        index=index,
    )
