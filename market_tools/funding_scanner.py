"""Read-only crypto perpetual funding-rate and basis scanner.

Fetches public perpetual funding rates, mark prices, and spot prices from
Binance and Bybit. No API keys, no order paths, and no authentication are
used. Each run logs one row per exchange/symbol into its own SQLite database
(``data/funding_rates.db`` by default) using WAL mode.

A failure for one exchange/symbol (HTTP error, geographic block, parse error,
empty payload) never aborts the run: it is logged as an explicit FETCH_ERROR
row and the remaining pairs continue.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
BINANCE_SPOT_BASE_URL = "https://api.binance.com"
BYBIT_BASE_URL = "https://api.bybit.com"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
DEFAULT_FUNDING_INTERVAL_HOURS = 8.0
HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "funding_rates.db"
SQLITE_BUSY_TIMEOUT_MS = 5000
DB_FILE_MODE = 0o600

STATUS_OK = "OK"
STATUS_FETCH_ERROR = "FETCH_ERROR"

TABLE_COLUMNS = [
    "id",
    "timestamp_utc",
    "exchange",
    "symbol",
    "funding_rate",
    "funding_interval_hours",
    "annualized_funding_apr_pct",
    "spot_price",
    "perp_price",
    "basis_pct",
    "status",
    "error_message",
]

HttpFetcher = Callable[[str], Any]


@dataclass(frozen=True)
class FundingSnapshot:
    timestamp_utc: str
    exchange: str
    symbol: str
    funding_rate: float | None
    funding_interval_hours: float
    annualized_funding_apr_pct: float | None
    spot_price: float | None
    perp_price: float | None
    basis_pct: float | None
    status: str
    error_message: str = ""


def _utc_now(timestamp_utc: datetime | str | None = None) -> str:
    if timestamp_utc is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(timestamp_utc, str):
        return timestamp_utc
    elif timestamp_utc.tzinfo is None:
        moment = timestamp_utc.replace(tzinfo=timezone.utc)
    else:
        moment = timestamp_utc.astimezone(timezone.utc)
    return moment.isoformat(timespec="seconds")


def _http_get_json(url: str) -> Any:
    """GET a public JSON endpoint with a 10-second timeout."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "crypto-funding-scanner/1.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else None


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _as_positive_float(value: Any, field: str) -> float:
    number = _as_float(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be positive, got {number!r}")
    return number


def annualized_funding_apr_pct(
    funding_rate: float,
    funding_interval_hours: float = DEFAULT_FUNDING_INTERVAL_HOURS,
) -> float:
    """Annualize a per-interval funding rate into an APR percentage."""
    if funding_interval_hours <= 0:
        raise ValueError("funding_interval_hours must be positive")
    return funding_rate * (24.0 / funding_interval_hours) * 365.0 * 100.0


def basis_pct(perp_price: float, spot_price: float) -> float:
    """Perpetual premium (or discount) versus spot, in percent."""
    if spot_price <= 0:
        raise ValueError("spot_price must be positive")
    return ((perp_price - spot_price) / spot_price) * 100.0


def cross_venue_funding_spread_apr_pct(
    binance: FundingSnapshot | None,
    bybit: FundingSnapshot | None,
) -> float | None:
    """Annualized Binance-minus-Bybit funding spread in percentage points.

    Returns ``None`` when either venue failed or has no usable APR for the
    same symbol.
    """
    if binance is None or bybit is None:
        return None
    if binance.symbol != bybit.symbol:
        raise ValueError("cross-venue spread requires the same symbol")
    if binance.status != STATUS_OK or bybit.status != STATUS_OK:
        return None
    binance_apr = binance.annualized_funding_apr_pct
    bybit_apr = bybit.annualized_funding_apr_pct
    if binance_apr is None or bybit_apr is None:
        return None
    return binance_apr - bybit_apr


def _error_snapshot(
    exchange: str,
    symbol: str,
    funding_interval_hours: float,
    timestamp_utc: str,
    error: BaseException,
) -> FundingSnapshot:
    return FundingSnapshot(
        timestamp_utc=timestamp_utc,
        exchange=exchange,
        symbol=symbol,
        funding_rate=None,
        funding_interval_hours=funding_interval_hours,
        annualized_funding_apr_pct=None,
        spot_price=None,
        perp_price=None,
        basis_pct=None,
        status=STATUS_FETCH_ERROR,
        error_message=f"{type(error).__name__}: {error}",
    )


def _binance_premium_row(payload: Any, symbol: str) -> Mapping[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping) and item.get("symbol") == symbol:
                return item
        raise ValueError(f"Binance premiumIndex has no row for {symbol}")
    if not isinstance(payload, Mapping):
        raise ValueError("Unexpected Binance premiumIndex response")
    if payload.get("symbol") not in (None, symbol):
        raise ValueError(
            f"Binance premiumIndex returned {payload.get('symbol')!r}, expected {symbol!r}"
        )
    return payload


def fetch_binance_snapshot(
    symbol: str,
    funding_interval_hours: float = DEFAULT_FUNDING_INTERVAL_HOURS,
    http_get: HttpFetcher | None = None,
    timestamp_utc: datetime | str | None = None,
) -> FundingSnapshot:
    """Fetch Binance USDT-perpetual funding, mark price, and spot price."""
    fetcher = http_get or _http_get_json
    stamp = _utc_now(timestamp_utc)
    try:
        premium = _binance_premium_row(
            fetcher(
                f"{BINANCE_FAPI_BASE_URL}/fapi/v1/premiumIndex"
                f"?symbol={urllib.parse.quote(symbol)}"
            ),
            symbol,
        )
        spot_payload = fetcher(
            f"{BINANCE_SPOT_BASE_URL}/api/v3/ticker/price"
            f"?symbol={urllib.parse.quote(symbol)}"
        )
        if not isinstance(spot_payload, Mapping):
            raise ValueError("Unexpected Binance spot ticker response")

        funding_rate = _as_float(premium.get("lastFundingRate"), "lastFundingRate")
        perp_price = _as_positive_float(premium.get("markPrice"), "markPrice")
        spot_price = _as_positive_float(spot_payload.get("price"), "price")
    except Exception as exc:
        return _error_snapshot(
            "binance", symbol, funding_interval_hours, stamp, exc
        )

    return FundingSnapshot(
        timestamp_utc=stamp,
        exchange="binance",
        symbol=symbol,
        funding_rate=funding_rate,
        funding_interval_hours=funding_interval_hours,
        annualized_funding_apr_pct=annualized_funding_apr_pct(
            funding_rate, funding_interval_hours
        ),
        spot_price=spot_price,
        perp_price=perp_price,
        basis_pct=basis_pct(perp_price, spot_price),
        status=STATUS_OK,
    )


def _bybit_ticker_row(payload: Any, venue: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unexpected Bybit {venue} response")
    ret_code = payload.get("retCode")
    if ret_code not in (0, "0"):
        raise RuntimeError(
            f"Bybit {venue} error retCode={ret_code!r}: {payload.get('retMsg')!r}"
        )
    result = payload.get("result")
    rows = result.get("list") if isinstance(result, Mapping) else None
    if not rows:
        raise ValueError(f"Bybit {venue} ticker list is empty")
    row = rows[0]
    if not isinstance(row, Mapping):
        raise ValueError(f"Unexpected Bybit {venue} ticker row")
    return row


def fetch_bybit_snapshot(
    symbol: str,
    funding_interval_hours: float = DEFAULT_FUNDING_INTERVAL_HOURS,
    http_get: HttpFetcher | None = None,
    timestamp_utc: datetime | str | None = None,
) -> FundingSnapshot:
    """Fetch Bybit USDT-perpetual funding, mark price, and spot price."""
    fetcher = http_get or _http_get_json
    stamp = _utc_now(timestamp_utc)
    try:
        perp = _bybit_ticker_row(
            fetcher(
                f"{BYBIT_BASE_URL}/v5/market/tickers"
                f"?category=linear&symbol={urllib.parse.quote(symbol)}"
            ),
            "linear",
        )
        spot = _bybit_ticker_row(
            fetcher(
                f"{BYBIT_BASE_URL}/v5/market/tickers"
                f"?category=spot&symbol={urllib.parse.quote(symbol)}"
            ),
            "spot",
        )
        funding_rate = _as_float(perp.get("fundingRate"), "fundingRate")
        perp_price = _as_positive_float(perp.get("markPrice"), "markPrice")
        spot_price = _as_positive_float(spot.get("lastPrice"), "lastPrice")
    except Exception as exc:
        return _error_snapshot(
            "bybit", symbol, funding_interval_hours, stamp, exc
        )

    return FundingSnapshot(
        timestamp_utc=stamp,
        exchange="bybit",
        symbol=symbol,
        funding_rate=funding_rate,
        funding_interval_hours=funding_interval_hours,
        annualized_funding_apr_pct=annualized_funding_apr_pct(
            funding_rate, funding_interval_hours
        ),
        spot_price=spot_price,
        perp_price=perp_price,
        basis_pct=basis_pct(perp_price, spot_price),
        status=STATUS_OK,
    )


def collect_snapshots(
    symbols: Sequence[str] | None = None,
    funding_interval_hours: float = DEFAULT_FUNDING_INTERVAL_HOURS,
    http_get: HttpFetcher | None = None,
    timestamp_utc: datetime | str | None = None,
) -> list[FundingSnapshot]:
    """Collect Binance and Bybit snapshots for every requested symbol."""
    requested = list(symbols or DEFAULT_SYMBOLS)
    stamp = _utc_now(timestamp_utc)
    snapshots: list[FundingSnapshot] = []
    for raw_symbol in requested:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        for exchange, fetcher in (
            ("binance", fetch_binance_snapshot),
            ("bybit", fetch_bybit_snapshot),
        ):
            try:
                snapshots.append(
                    fetcher(
                        symbol,
                        funding_interval_hours=funding_interval_hours,
                        http_get=http_get,
                        timestamp_utc=stamp,
                    )
                )
            except Exception as exc:
                snapshots.append(
                    _error_snapshot(
                        exchange, symbol, funding_interval_hours, stamp, exc
                    )
                )
    return snapshots


def connect_funding_db(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        Path(db_path),
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
    )
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def init_funding_db(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_funding_db(path)
    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    funding_rate REAL,
                    funding_interval_hours REAL NOT NULL,
                    annualized_funding_apr_pct REAL,
                    spot_price REAL,
                    perp_price REAL,
                    basis_pct REAL,
                    status TEXT NOT NULL,
                    error_message TEXT NOT NULL DEFAULT ''
                )
                """
            )
    finally:
        connection.close()
    os.chmod(path, DB_FILE_MODE)
    return path


def log_snapshots(
    snapshots: Iterable[FundingSnapshot],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert snapshots into the dedicated funding_rates database."""
    path = init_funding_db(db_path)
    rows = [
        (
            snapshot.timestamp_utc,
            snapshot.exchange,
            snapshot.symbol,
            snapshot.funding_rate,
            snapshot.funding_interval_hours,
            snapshot.annualized_funding_apr_pct,
            snapshot.spot_price,
            snapshot.perp_price,
            snapshot.basis_pct,
            snapshot.status,
            snapshot.error_message,
        )
        for snapshot in snapshots
    ]
    if rows:
        connection = connect_funding_db(path)
        try:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO funding_rates(
                        timestamp_utc,
                        exchange,
                        symbol,
                        funding_rate,
                        funding_interval_hours,
                        annualized_funding_apr_pct,
                        spot_price,
                        perp_price,
                        basis_pct,
                        status,
                        error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        finally:
            connection.close()
    os.chmod(path, DB_FILE_MODE)
    return len(rows)


def load_funding_rates(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read logged rows back, newest last (or capped by ``limit``)."""
    path = Path(db_path)
    if not path.exists():
        return []
    connection = connect_funding_db(path)
    try:
        query = (
            "SELECT id, timestamp_utc, exchange, symbol, funding_rate, "
            "funding_interval_hours, annualized_funding_apr_pct, spot_price, "
            "perp_price, basis_pct, status, error_message FROM funding_rates "
            "ORDER BY id"
        )
        if limit is not None:
            query += " DESC LIMIT ?"
            rows = connection.execute(query, (int(limit),)).fetchall()
            rows = list(reversed(rows))
        else:
            rows = connection.execute(query).fetchall()
    finally:
        connection.close()
    return [dict(zip(TABLE_COLUMNS, row)) for row in rows]


def _format_summary(snapshots: Iterable[FundingSnapshot]) -> str:
    rows = list(snapshots)
    header = (
        f"{'symbol':<8} {'exchange':<8} {'status':<12} {'funding':>11} "
        f"{'APR':>10} {'spot':>14} {'perp':>14} {'basis':>10}"
    )
    lines = [
        "Crypto Funding Rate & Basis Scanner (read-only)",
        header,
        "-" * len(header),
    ]
    for snapshot in rows:
        if snapshot.status == STATUS_OK:
            funding = f"{snapshot.funding_rate * 100:+.4f}%"
            apr = f"{snapshot.annualized_funding_apr_pct:+.2f}%"
            spot = f"{snapshot.spot_price:,.2f}"
            perp = f"{snapshot.perp_price:,.2f}"
            basis = f"{snapshot.basis_pct:+.3f}%"
        else:
            funding = apr = spot = perp = basis = "—"
        lines.append(
            f"{snapshot.symbol:<8} {snapshot.exchange:<8} "
            f"{snapshot.status:<12} {funding:>11} {apr:>10} {spot:>14} "
            f"{perp:>14} {basis:>10}"
        )

    lines.append("")
    lines.append("Cross-venue funding APR spread (Binance - Bybit):")
    by_symbol: dict[str, dict[str, FundingSnapshot]] = {}
    for snapshot in rows:
        by_symbol.setdefault(snapshot.symbol, {})[snapshot.exchange] = snapshot
    for symbol in sorted(by_symbol):
        venues = by_symbol[symbol]
        spread = cross_venue_funding_spread_apr_pct(
            venues.get("binance"), venues.get("bybit")
        )
        if spread is None:
            lines.append(f"  {symbol}: n/a")
        else:
            lines.append(f"  {symbol}: {spread:+.2f} pp")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch public Binance/Bybit funding, mark, and spot prices, "
            "log them to a dedicated SQLite database, and print a summary."
        )
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Symbols to scan (default: BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--funding-interval-hours",
        type=float,
        default=DEFAULT_FUNDING_INTERVAL_HOURS,
        help="Funding interval in hours (default: 8.0)",
    )
    return parser


def main(
    argv: list[str] | None = None,
    http_get: HttpFetcher | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    snapshots = collect_snapshots(
        symbols=args.symbols,
        funding_interval_hours=args.funding_interval_hours,
        http_get=http_get,
    )
    logged = log_snapshots(snapshots, args.db_path)
    print(_format_summary(snapshots))
    print(f"\nLogged {logged} row(s) to {args.db_path}")
    return 0 if any(snapshot.status == STATUS_OK for snapshot in snapshots) else 1


if __name__ == "__main__":
    raise SystemExit(main())
