"""Independent, read-only reconciliation for the discipline portfolio.

This module is deliberately standalone: it imports nothing from
``discipline_bot`` or ``momentum_bot.rebalance`` (or any other trading bot
code). It talks to Alpaca's paper REST API directly using the standard
library and performs only GET requests.

Checks:
- SPY / TLT weights versus the 80% / 20% target, flagging drift beyond 3%.
- Unexpected or unauthorized symbols held in the account.
- Fill status for the current US Eastern ISO week
  (``discipline_YYYYWww``, including ``_rN`` / ``_sell`` suffixes).

Exit code is 0 when clean and 1 when any anomaly is detected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

PAPER_API_BASE_URL = "https://paper-api.alpaca.markets"
PAPER_API_HOST = "paper-api.alpaca.markets"
SPY_TARGET = 0.80
TLT_TARGET = 0.20
DRIFT_BAND = 0.03
ALLOWED_SYMBOLS = frozenset({"SPY", "TLT"})
EASTERN_TZ = ZoneInfo("America/New_York")
HTTP_TIMEOUT_SECONDS = 15.0
_QUANTITY_EPSILON = 1e-9
_DRIFT_EPSILON = 1e-9

HttpGetter = Callable[[str, Mapping[str, str]], Any]


@dataclass(frozen=True)
class Finding:
    code: str
    message: str


@dataclass(frozen=True)
class ReconciliationResult:
    week_id: str
    spy_weight: float | None
    tlt_weight: float | None
    fill_client_order_ids: tuple[str, ...]
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings


def assert_paper_endpoint(base_url: str = PAPER_API_BASE_URL) -> str:
    """Hard-assert that an Alpaca base URL points at the paper endpoint."""
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" or parsed.hostname != PAPER_API_HOST:
        raise ValueError(
            "Refusing non-paper Alpaca endpoint "
            f"{base_url!r}; paper-only host is {PAPER_API_BASE_URL}."
        )
    return PAPER_API_BASE_URL


def resolve_paper_base_url(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return assert_paper_endpoint(env.get("ALPACA_BASE_URL") or PAPER_API_BASE_URL)


def _credentials(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if environ is None else environ
    key = (
        env.get("ALPACA_API_KEY")
        or env.get("APCA_API_KEY_ID")
        or env.get("ALPACA_KEY")
    )
    secret = (
        env.get("ALPACA_API_SECRET")
        or env.get("APCA_API_SECRET_KEY")
        or env.get("ALPACA_SECRET")
    )
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca API credentials. Set ALPACA_API_KEY and "
            "ALPACA_API_SECRET in the environment."
        )
    return key, secret


def _http_get_json(url: str, headers: Mapping[str, str]) -> Any:
    request = urllib.request.Request(
        url,
        headers=dict(headers),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else None


class AlpacaPaperClient:
    """Minimal read-only Alpaca paper client built on urllib."""

    def __init__(
        self,
        key: str | None = None,
        secret: str | None = None,
        base_url: str | None = None,
        http_get: HttpGetter | None = None,
    ) -> None:
        self.base_url = assert_paper_endpoint(
            base_url or resolve_paper_base_url()
        )
        if key is None or secret is None:
            env_key, env_secret = _credentials()
            key = key or env_key
            secret = secret or env_secret
        self._headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        }
        self._http_get = http_get or _http_get_json

    def _get(self, path: str, params: Mapping[str, str] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        assert_paper_endpoint(url)
        return self._http_get(url, self._headers)

    def get_positions(self) -> list[Mapping[str, Any]]:
        payload = self._get("/v2/positions")
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected /v2/positions response from Alpaca.")
        return [item for item in payload if isinstance(item, Mapping)]

    def get_fill_activities(self) -> list[Mapping[str, Any]]:
        payload = self._get(
            "/v2/account/activities",
            {"activity_types": "FILL"},
        )
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected /v2/account/activities response from Alpaca.")
        return [item for item in payload if isinstance(item, Mapping)]


def current_week_id(now: datetime | None = None) -> str:
    current = now or datetime.now(EASTERN_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=EASTERN_TZ)
    iso_year, iso_week, _ = current.astimezone(EASTERN_TZ).isocalendar()
    return f"discipline_{iso_year}W{iso_week:02d}"


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _market_value(position: Mapping[str, Any]) -> float:
    value = _to_float(position.get("market_value"))
    if value is not None:
        return value
    quantity = _to_float(position.get("qty")) or 0.0
    price = _to_float(position.get("current_price")) or 0.0
    return quantity * price


def analyze_positions(
    positions: list[Mapping[str, Any]],
) -> tuple[float | None, float | None, list[Finding]]:
    findings: list[Finding] = []
    values: dict[str, float] = {}

    for position in positions:
        symbol = str(position.get("symbol") or "").strip().upper()
        quantity = _to_float(position.get("qty")) or 0.0
        if not symbol or abs(quantity) <= _QUANTITY_EPSILON:
            continue
        if symbol not in ALLOWED_SYMBOLS:
            findings.append(
                Finding(
                    "UNAUTHORIZED_SYMBOL",
                    f"Unexpected symbol held: {symbol} (qty {quantity:g}).",
                )
            )
        elif quantity < 0:
            findings.append(
                Finding(
                    "SHORT_POSITION",
                    f"{symbol} is short (qty {quantity:g}); discipline "
                    "portfolio is long-only.",
                )
            )
        values[symbol] = values.get(symbol, 0.0) + abs(_market_value(position))

    total = sum(values.values())
    if total <= 0:
        findings.append(
            Finding(
                "NO_POSITIONS",
                "No invested positions found; cannot verify the 80/20 weights.",
            )
        )
        return None, None, findings

    spy_weight = values.get("SPY", 0.0) / total
    tlt_weight = values.get("TLT", 0.0) / total
    for symbol, weight, target in (
        ("SPY", spy_weight, SPY_TARGET),
        ("TLT", tlt_weight, TLT_TARGET),
    ):
        drift = weight - target
        if abs(drift) > DRIFT_BAND + _DRIFT_EPSILON:
            findings.append(
                Finding(
                    "WEIGHT_DRIFT",
                    f"{symbol} weight {weight:.2%} is {drift:+.2%} from the "
                    f"{target:.0%} target (band ±{DRIFT_BAND:.0%}).",
                )
            )
    return spy_weight, tlt_weight, findings


def check_week_fill(
    activities: list[Mapping[str, Any]],
    week_id: str,
) -> tuple[list[str], list[Finding]]:
    matched: set[str] = set()
    for activity in activities:
        activity_type = str(activity.get("activity_type") or "").strip().upper()
        if activity_type != "FILL":
            continue
        client_order_id = activity.get("client_order_id")
        if not isinstance(client_order_id, str):
            continue
        if client_order_id == week_id or client_order_id.startswith(f"{week_id}_"):
            matched.add(client_order_id)

    findings: list[Finding] = []
    if not matched:
        findings.append(
            Finding(
                "MISSING_WEEK_FILL",
                f"No fill found for {week_id} in the paper account.",
            )
        )
    return sorted(matched), findings


def reconcile(
    client: AlpacaPaperClient,
    now: datetime | None = None,
) -> ReconciliationResult:
    week_id = current_week_id(now)
    spy_weight, tlt_weight, findings = analyze_positions(client.get_positions())
    fill_ids, fill_findings = check_week_fill(
        client.get_fill_activities(),
        week_id,
    )
    findings.extend(fill_findings)
    return ReconciliationResult(
        week_id=week_id,
        spy_weight=spy_weight,
        tlt_weight=tlt_weight,
        fill_client_order_ids=tuple(fill_ids),
        findings=tuple(findings),
    )


def _format_weight(weight: float | None) -> str:
    return f"{weight:.2%}" if weight is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only reconciliation of the paper discipline portfolio: "
            "80/20 weights, authorized symbols, and this ISO week's fill."
        )
    )
    parser.parse_args(argv)

    try:
        result = reconcile(AlpacaPaperClient())
    except (RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Week: {result.week_id}")
    print(f"SPY weight: {_format_weight(result.spy_weight)} (target 80%)")
    print(f"TLT weight: {_format_weight(result.tlt_weight)} (target 20%)")
    if result.fill_client_order_ids:
        print("Week fills: " + ", ".join(result.fill_client_order_ids))
    else:
        print("Week fills: none")

    if result.ok:
        print("OK: reconciliation clean.")
        return 0

    for finding in result.findings:
        print(f"ANOMALY [{finding.code}] {finding.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
