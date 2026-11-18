"""Read-only SPY Gamma Exposure (GEX) calculator and daily snapshot logger.

The model is intentionally simple and observational:

- Pull the current SPY spot price and the nearest non-expired option chain
  from yfinance.
- Keep only rows with positive open interest and positive implied volatility.
- Compute Black-Scholes gamma for each strike at the current spot, then
  aggregate net dealer gamma exposure as::

      net_gex = (call_oi * call_gamma - put_oi * put_gamma) * 100 * S**2 * 0.01

- Derive the call wall (max call OI), put wall (max put OI), and gamma flip
  (the interpolated spot where total net GEX crosses from negative to
  positive).

Embedded dealer-positioning assumption (not observed data):
    Net GEX = Call GEX - Put GEX treats dealers as net long calls (positive,
    stabilizing gamma) and net short puts (negative, amplifying gamma), the
    standard index-put-hedging convention. Open interest alone does not reveal
    true dealer direction: calls and puts can be held by either side of the
    trade. The sign is therefore an assumption, not measured dealer inventory.

No broker APIs, order paths, or API keys are used. The only write is the
deduplicated daily snapshot CSV under ``data/gex_history.csv``. Running
``python -m market_tools.gex_calculator`` collects and logs one snapshot for
headless weekday cron/GitHub Actions use.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN_TZ = ZoneInfo("America/New_York")
RISK_FREE_RATE = 0.045
CONTRACT_MULTIPLIER = 100.0
GEX_NOTIONAL_SCALE = 0.01
MIN_TIME_TO_EXPIRY = 1.0 / (365.0 * 24.0)
GAMMA_GRID_LOWER = 0.90
GAMMA_GRID_UPPER = 1.10
GAMMA_GRID_STEPS = 81
DEFAULT_SYMBOL = "SPY"
DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "gex_history.csv"

STATUS_OK = "OK"
STATUS_FETCH_ERROR = "FETCH_ERROR"
HISTORY_COLUMNS = [
    "date",
    "timestamp_et",
    "expiration",
    "spot",
    "call_wall",
    "put_wall",
    "gamma_flip",
    "net_gex_total",
    "zero_dte_net_gex",
    "put_call_oi_ratio",
    "max_pain",
    "dealer_gamma_flip_dist",
    "status",
    "error_message",
]


@dataclass(frozen=True)
class StrikeGex:
    """Per-strike inputs and current-spot net GEX."""

    strike: float
    call_oi: float
    put_oi: float
    call_iv: float
    put_iv: float
    call_gamma: float
    put_gamma: float
    net_gex: float


@dataclass
class GexInputs:
    """Raw inputs fetched from yfinance."""

    symbol: str
    spot: float
    expiration: str
    calls: Any
    puts: Any
    zero_dte_calls: Any = None
    zero_dte_puts: Any = None


@dataclass
class GexSnapshot:
    """One observational GEX snapshot, including explicit error states."""

    symbol: str
    date_et: str
    timestamp_et: str
    expiration: str | None
    spot: float | None
    call_wall: float | None
    put_wall: float | None
    gamma_flip: float | None
    net_gex_total: float | None
    status: str
    error_message: str = ""
    zero_dte_net_gex: float | None = None
    put_call_oi_ratio: float | None = None
    max_pain: float | None = None
    dealer_gamma_flip_dist: float | None = None
    strikes: list[StrikeGex] = field(default_factory=list)
    risk_free_rate: float = RISK_FREE_RATE


def _et_now(now_et: datetime | None = None) -> datetime:
    if now_et is None:
        return datetime.now(EASTERN_TZ)
    if now_et.tzinfo is None:
        return now_et.replace(tzinfo=EASTERN_TZ)
    return now_et.astimezone(EASTERN_TZ)


def norm_pdf(x: float) -> float:
    """Standard normal probability density, without SciPy."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def time_to_expiry_years(days_to_expiry: float) -> float:
    """Convert calendar days to years, flooring at one hour to avoid 0DTE / 0."""
    return max(float(days_to_expiry) / 365.0, MIN_TIME_TO_EXPIRY)


def days_to_expiry(expiration: str | date, now_et: datetime | None = None) -> int:
    """Calendar days from the US Eastern date to the expiration date (never negative)."""
    if isinstance(expiration, str):
        expiration_date = date.fromisoformat(expiration[:10])
    else:
        expiration_date = expiration
    return max((expiration_date - _et_now(now_et).date()).days, 0)


def black_scholes_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    implied_volatility: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Black-Scholes gamma for a European option."""
    if (
        spot <= 0
        or strike <= 0
        or time_to_expiry <= 0
        or implied_volatility <= 0
        or not all(
            math.isfinite(value)
            for value in (spot, strike, time_to_expiry, implied_volatility, risk_free_rate)
        )
    ):
        return 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) + (risk_free_rate + 0.5 * implied_volatility**2) * time_to_expiry
    ) / (implied_volatility * sqrt_t)
    return norm_pdf(d1) / (spot * implied_volatility * sqrt_t)


def _clean_chain(frame: Any) -> pd.DataFrame:
    """Filter invalid option rows and normalize numeric columns."""
    columns = ["strike", "openInterest", "impliedVolatility"]
    if frame is None:
        return pd.DataFrame(columns=columns)
    try:
        working = frame.loc[:, columns].copy()
    except Exception:
        return pd.DataFrame(columns=columns)

    for column in columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=columns)
    working = working[
        (working["strike"] > 0)
        & (working["openInterest"] > 0)
        & (working["impliedVolatility"] > 0)
    ]
    return working


def _aggregate_chain(frame: Any) -> dict[float, tuple[float, float]]:
    """Aggregate duplicate strike rows into (open interest, OI-weighted IV)."""
    clean = _clean_chain(frame)
    aggregated: dict[float, tuple[float, float]] = {}
    if clean.empty:
        return aggregated

    for strike, group in clean.groupby("strike", sort=True):
        open_interest = float(group["openInterest"].sum())
        if open_interest <= 0:
            continue
        weighted_iv = float(
            (group["impliedVolatility"] * group["openInterest"]).sum() / open_interest
        )
        aggregated[float(strike)] = (open_interest, weighted_iv)
    return aggregated


def calculate_strike_gex(
    spot: float,
    calls: Any,
    puts: Any,
    days_until_expiry: int | float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> list[StrikeGex]:
    """Compute per-strike Net GEX at the current spot.

    Uses Net GEX = Call GEX - Put GEX, which assumes dealers are net long
    calls (stabilizing) and net short puts (amplifying). Open interest does
    not reveal true dealer positioning; this is a convention, not an observed
    inventory.
    """
    if spot <= 0 or not math.isfinite(spot):
        raise ValueError("Spot price must be a positive finite number.")

    time_to_expiry = time_to_expiry_years(days_until_expiry)
    call_map = _aggregate_chain(calls)
    put_map = _aggregate_chain(puts)
    if not call_map and not put_map:
        raise ValueError("No valid option rows remain after filtering.")

    rows: list[StrikeGex] = []
    for strike in sorted(set(call_map) | set(put_map)):
        call_oi, call_iv = call_map.get(strike, (0.0, 0.0))
        put_oi, put_iv = put_map.get(strike, (0.0, 0.0))
        call_gamma = (
            black_scholes_gamma(spot, strike, time_to_expiry, call_iv, risk_free_rate)
            if call_oi > 0
            else 0.0
        )
        put_gamma = (
            black_scholes_gamma(spot, strike, time_to_expiry, put_iv, risk_free_rate)
            if put_oi > 0
            else 0.0
        )
        net_gex = (
            (call_oi * call_gamma - put_oi * put_gamma)
            * CONTRACT_MULTIPLIER
            * spot**2
            * GEX_NOTIONAL_SCALE
        )
        rows.append(
            StrikeGex(
                strike=strike,
                call_oi=call_oi,
                put_oi=put_oi,
                call_iv=call_iv,
                put_iv=put_iv,
                call_gamma=call_gamma,
                put_gamma=put_gamma,
                net_gex=net_gex,
            )
        )
    return rows


def find_call_wall(strikes: Iterable[StrikeGex]) -> float | None:
    """Strike with the largest call open interest."""
    candidates = [row for row in strikes if row.call_oi > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.call_oi).strike


def find_put_wall(strikes: Iterable[StrikeGex]) -> float | None:
    """Strike with the largest put open interest."""
    candidates = [row for row in strikes if row.put_oi > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.put_oi).strike


def find_max_pain(strikes: Iterable[StrikeGex]) -> float | None:
    """Strike with the largest total (call + put) open interest."""
    candidates = [
        row for row in strikes if row.call_oi + row.put_oi > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.call_oi + row.put_oi).strike


def calculate_put_call_oi_ratio(strikes: Iterable[StrikeGex]) -> float | None:
    """Total put open interest divided by total call open interest."""
    rows = list(strikes)
    call_oi = sum(row.call_oi for row in rows)
    put_oi = sum(row.put_oi for row in rows)
    if call_oi <= 0:
        return None
    return put_oi / call_oi


def gamma_flip_distance(
    spot: float | None,
    gamma_flip: float | None,
) -> float | None:
    """Absolute distance between spot and the gamma flip level."""
    if spot is None or gamma_flip is None:
        return None
    if not (math.isfinite(spot) and math.isfinite(gamma_flip)):
        return None
    return abs(float(spot) - float(gamma_flip))


def calculate_zero_dte_net_gex(
    spot: float,
    calls: Any,
    puts: Any,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float | None:
    """Net GEX for options expiring today.

    Returns ``None`` when no valid same-day rows are available.
    """
    try:
        zero_dte_strikes = calculate_strike_gex(
            spot=spot,
            calls=calls,
            puts=puts,
            days_until_expiry=0,
            risk_free_rate=risk_free_rate,
        )
    except ValueError:
        return None
    return float(sum(row.net_gex for row in zero_dte_strikes))


def aggregate_net_gex_at_spot(
    strikes: Iterable[StrikeGex],
    candidate_spot: float,
    time_to_expiry: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Total Net GEX at a hypothetical spot, holding each strike's OI and IV fixed."""
    if candidate_spot <= 0 or not math.isfinite(candidate_spot):
        return 0.0

    total = 0.0
    for row in strikes:
        call_gamma = (
            black_scholes_gamma(
                candidate_spot, row.strike, time_to_expiry, row.call_iv, risk_free_rate
            )
            if row.call_oi > 0
            else 0.0
        )
        put_gamma = (
            black_scholes_gamma(
                candidate_spot, row.strike, time_to_expiry, row.put_iv, risk_free_rate
            )
            if row.put_oi > 0
            else 0.0
        )
        total += (
            (row.call_oi * call_gamma - row.put_oi * put_gamma)
            * CONTRACT_MULTIPLIER
            * candidate_spot**2
            * GEX_NOTIONAL_SCALE
        )
    return total


def find_gamma_flip(
    strikes: Iterable[StrikeGex],
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float = RISK_FREE_RATE,
    grid_lower: float = GAMMA_GRID_LOWER,
    grid_upper: float = GAMMA_GRID_UPPER,
    grid_steps: int = GAMMA_GRID_STEPS,
) -> float | None:
    """Interpolate the spot where total Net GEX crosses from negative to positive.

    Returns ``None`` when no such crossing exists inside the candidate grid.
    """
    rows = list(strikes)
    if (
        not rows
        or spot <= 0
        or not math.isfinite(spot)
        or time_to_expiry <= 0
        or not math.isfinite(time_to_expiry)
        or grid_steps < 2
    ):
        return None

    lower = grid_lower * spot
    upper = grid_upper * spot
    if lower <= 0 or upper <= lower:
        return None

    grid = [
        lower + (upper - lower) * index / (grid_steps - 1)
        for index in range(grid_steps)
    ]
    totals = [
        aggregate_net_gex_at_spot(rows, candidate, time_to_expiry, risk_free_rate)
        for candidate in grid
    ]

    crossings: list[float] = []
    for index in range(1, len(grid)):
        previous_total = totals[index - 1]
        current_total = totals[index]
        if previous_total < 0 <= current_total:
            denominator = current_total - previous_total
            if denominator == 0:
                crossings.append(grid[index])
            else:
                fraction = (0.0 - previous_total) / denominator
                crossings.append(grid[index - 1] + fraction * (grid[index] - grid[index - 1]))

    if not crossings:
        return None
    return min(crossings, key=lambda candidate: abs(candidate - spot))


def select_nearest_expiration(
    expirations: Iterable[Any],
    now_et: datetime | None = None,
) -> str:
    """Pick the nearest expiration on or after today's US Eastern date."""
    today = _et_now(now_et).date()
    valid: list[tuple[date, str]] = []
    for raw in expirations or []:
        text = str(raw)[:10]
        try:
            expiration_date = date.fromisoformat(text)
        except ValueError:
            continue
        if expiration_date >= today:
            valid.append((expiration_date, text))
    if not valid:
        raise ValueError("yfinance returned no non-expired SPY option expirations.")
    return min(valid)[0].isoformat()


def fetch_spot_price(ticker: Any) -> float:
    """Fetch the current spot price from yfinance, with a history fallback."""
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        try:
            raw_price = fast_info["last_price"]
        except Exception:
            raw_price = getattr(fast_info, "last_price", None)
        try:
            price = float(raw_price)
            if math.isfinite(price) and price > 0:
                return price
        except (TypeError, ValueError):
            pass

    history = ticker.history(period="5d", interval="1d", auto_adjust=False)
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        raise ValueError("yfinance returned no SPY price history.")
    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if closes.empty:
        raise ValueError("yfinance returned no usable SPY closing prices.")
    price = float(closes.iloc[-1])
    if not math.isfinite(price) or price <= 0:
        raise ValueError("yfinance returned an invalid SPY spot price.")
    return price


def fetch_option_chain(ticker: Any, expiration: str) -> tuple[Any, Any]:
    """Fetch a non-empty calls/puts chain for one expiration."""
    chain = ticker.option_chain(expiration)
    calls = getattr(chain, "calls", None)
    puts = getattr(chain, "puts", None)
    if calls is None or puts is None or len(calls) == 0 or len(puts) == 0:
        raise ValueError(f"yfinance returned an empty option chain for {expiration}.")
    return calls, puts


def _default_ticker_factory(symbol: str) -> Any:
    import yfinance as yf

    return yf.Ticker(symbol)


def fetch_gex_inputs(
    symbol: str = DEFAULT_SYMBOL,
    ticker_factory: Callable[[str], Any] | None = None,
    now_et: datetime | None = None,
) -> GexInputs:
    """Fetch spot, nearest expiration, option chain, and today's 0DTE chain."""
    factory = ticker_factory or _default_ticker_factory
    now = _et_now(now_et)
    ticker = factory(symbol)
    spot = fetch_spot_price(ticker)
    raw_options = list(getattr(ticker, "options", ()) or ())
    expiration = select_nearest_expiration(raw_options, now_et=now)
    calls, puts = fetch_option_chain(ticker, expiration)

    today_text = now.date().isoformat()
    zero_dte_calls: Any = None
    zero_dte_puts: Any = None
    if expiration == today_text:
        zero_dte_calls, zero_dte_puts = calls, puts
    elif any(str(raw)[:10] == today_text for raw in raw_options):
        try:
            zero_dte_calls, zero_dte_puts = fetch_option_chain(ticker, today_text)
        except ValueError:
            zero_dte_calls, zero_dte_puts = None, None

    return GexInputs(
        symbol=symbol,
        spot=spot,
        expiration=expiration,
        calls=calls,
        puts=puts,
        zero_dte_calls=zero_dte_calls,
        zero_dte_puts=zero_dte_puts,
    )


def calculate_gex_snapshot(
    spot: float,
    expiration: str,
    calls: Any,
    puts: Any,
    symbol: str = DEFAULT_SYMBOL,
    now_et: datetime | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
    zero_dte_calls: Any | None = None,
    zero_dte_puts: Any | None = None,
) -> GexSnapshot:
    """Calculate one fully-populated GEX snapshot from already-fetched inputs."""
    now = _et_now(now_et)
    expiry_days = days_to_expiry(expiration, now)
    time_to_expiry = time_to_expiry_years(expiry_days)
    strikes = calculate_strike_gex(
        spot=spot,
        calls=calls,
        puts=puts,
        days_until_expiry=expiry_days,
        risk_free_rate=risk_free_rate,
    )
    call_wall = find_call_wall(strikes)
    put_wall = find_put_wall(strikes)
    gamma_flip = find_gamma_flip(
        strikes,
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
    )
    net_gex_total = float(sum(row.net_gex for row in strikes))

    if (
        zero_dte_calls is None
        and zero_dte_puts is None
        and str(expiration)[:10] == now.date().isoformat()
    ):
        zero_dte_calls, zero_dte_puts = calls, puts
    zero_dte_net_gex = (
        calculate_zero_dte_net_gex(
            spot=spot,
            calls=zero_dte_calls,
            puts=zero_dte_puts,
            risk_free_rate=risk_free_rate,
        )
        if zero_dte_calls is not None or zero_dte_puts is not None
        else None
    )

    return GexSnapshot(
        symbol=symbol,
        date_et=now.date().isoformat(),
        timestamp_et=now.isoformat(timespec="seconds"),
        expiration=expiration,
        spot=float(spot),
        call_wall=call_wall,
        put_wall=put_wall,
        gamma_flip=gamma_flip,
        net_gex_total=net_gex_total,
        status=STATUS_OK,
        zero_dte_net_gex=zero_dte_net_gex,
        put_call_oi_ratio=calculate_put_call_oi_ratio(strikes),
        max_pain=find_max_pain(strikes),
        dealer_gamma_flip_dist=gamma_flip_distance(spot, gamma_flip),
        strikes=strikes,
        risk_free_rate=risk_free_rate,
    )


def gamma_regime(net_gex_total: float | None) -> str:
    """Human-readable dealer-gamma regime label."""
    if net_gex_total is None:
        return "Unknown"
    return "Long Gamma (stabilizing)" if net_gex_total >= 0 else "Short Gamma (amplifying)"


def _format_csv_number(value: float | None) -> str:
    if value is None:
        return ""
    return repr(float(value))


def _snapshot_row(snapshot: GexSnapshot) -> dict[str, str]:
    return {
        "date": snapshot.date_et,
        "timestamp_et": snapshot.timestamp_et,
        "expiration": snapshot.expiration or "",
        "spot": _format_csv_number(snapshot.spot),
        "call_wall": _format_csv_number(snapshot.call_wall),
        "put_wall": _format_csv_number(snapshot.put_wall),
        "gamma_flip": _format_csv_number(snapshot.gamma_flip),
        "net_gex_total": _format_csv_number(snapshot.net_gex_total),
        "zero_dte_net_gex": _format_csv_number(snapshot.zero_dte_net_gex),
        "put_call_oi_ratio": _format_csv_number(snapshot.put_call_oi_ratio),
        "max_pain": _format_csv_number(snapshot.max_pain),
        "dealer_gamma_flip_dist": _format_csv_number(
            snapshot.dealer_gamma_flip_dist
        ),
        "status": snapshot.status,
        "error_message": snapshot.error_message,
    }


def load_gex_history(path: str | Path = DEFAULT_HISTORY_PATH) -> list[dict[str, str]]:
    """Read the snapshot CSV, returning an empty list when it does not exist."""
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def log_gex_snapshot(
    snapshot: GexSnapshot,
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> str:
    """Append or replace one date's snapshot, deduplicating by US Eastern date.

    Returns ``"appended"``, ``"replaced"``, or ``"skipped"``.
    """
    target = Path(path)
    rows = load_gex_history(target)
    new_row = _snapshot_row(snapshot)
    matching_indices = [
        index for index, row in enumerate(rows) if row.get("date") == snapshot.date_et
    ]

    action = "appended"
    if matching_indices:
        existing_statuses = {rows[index].get("status", "") for index in matching_indices}
        if STATUS_OK in existing_statuses:
            return "skipped"

        action = "replaced"
        first_index = matching_indices[0]
        rows = [row for index, row in enumerate(rows) if index not in matching_indices]
        rows.insert(first_index, new_row)
    else:
        rows.append(new_row)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    return action


def collect_gex_snapshot(
    symbol: str = DEFAULT_SYMBOL,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    ticker_factory: Callable[[str], Any] | None = None,
    now_et: datetime | None = None,
) -> GexSnapshot:
    """Fetch, calculate, and log one snapshot. Always writes an explicit error row."""
    now = _et_now(now_et)
    try:
        inputs = fetch_gex_inputs(symbol=symbol, ticker_factory=ticker_factory, now_et=now)
        snapshot = calculate_gex_snapshot(
            spot=inputs.spot,
            expiration=inputs.expiration,
            calls=inputs.calls,
            puts=inputs.puts,
            symbol=symbol,
            now_et=now,
            zero_dte_calls=inputs.zero_dte_calls,
            zero_dte_puts=inputs.zero_dte_puts,
        )
    except Exception as exc:
        snapshot = GexSnapshot(
            symbol=symbol,
            date_et=now.date().isoformat(),
            timestamp_et=now.isoformat(timespec="seconds"),
            expiration=None,
            spot=None,
            call_wall=None,
            put_wall=None,
            gamma_flip=None,
            net_gex_total=None,
            status=STATUS_FETCH_ERROR,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    log_gex_snapshot(snapshot, history_path)
    return snapshot


def main(argv: list[str] | None = None) -> int:
    """Collect and log one snapshot for headless weekday cron jobs."""
    parser = argparse.ArgumentParser(
        description="Collect and log one read-only SPY GEX snapshot."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH))
    args = parser.parse_args(argv)

    snapshot = collect_gex_snapshot(
        symbol=args.symbol,
        history_path=args.history_path,
    )
    print(
        f"{snapshot.symbol} GEX {snapshot.status} date={snapshot.date_et} "
        f"expiration={snapshot.expiration} spot={snapshot.spot} "
        f"net_gex_total={snapshot.net_gex_total} logged_to={args.history_path}"
    )
    if snapshot.status != STATUS_OK:
        print(f"GEX fetch error: {snapshot.error_message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
