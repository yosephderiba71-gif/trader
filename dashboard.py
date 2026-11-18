"""Streamlit dashboard for the two paper-trading accounts.

Discipline Bot reads ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET``. Trial #24 reads
``TRIAL24_ALPACA_API_KEY`` / ``TRIAL24_ALPACA_SECRET_KEY``. Credentials come
from the environment or Streamlit secrets, are never logged or rendered, and
are used only for read-only Alpaca account queries.

The compatibility CLI still delegates to ``market_viewer``. The optional Market
Viewer and SPY GEX sections remain available from the dashboard sidebar.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _payload() -> dict:
    from market_viewer import build_payload, default_provider

    return build_payload(default_provider())


def _format_gex_value(value: float | None) -> str:
    if value is None:
        return "—"
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:+,.2f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:+,.2f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:+,.2f}K"
    return f"{value:+,.2f}"


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value:+.2%}"


def _fetch_account_snapshot(api_key: str, api_secret: str) -> dict:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    client = TradingClient(api_key, api_secret, paper=True)
    account = client.get_account()
    positions = client.get_all_positions()
    orders = client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=25)
    )

    position_rows = []
    for position in positions or []:
        position_rows.append(
            {
                "Symbol": _text(getattr(position, "symbol", "")),
                "Side": _text(getattr(position, "side", "")).upper(),
                "Qty": _number(getattr(position, "qty", 0)),
                "Avg Entry": _money(_number(getattr(position, "avg_entry_price", 0))),
                "Current": _money(_number(getattr(position, "current_price", 0))),
                "Market Value": _money(_number(getattr(position, "market_value", 0))),
                "Unrealized P/L": _money(
                    _number(getattr(position, "unrealized_pl", 0))
                ),
                "Unrealized P/L %": _pct(
                    _number(getattr(position, "unrealized_plpc", 0))
                ),
            }
        )

    order_rows = []
    for order in sorted(
        orders or [],
        key=lambda item: _text(getattr(item, "submitted_at", "")),
        reverse=True,
    ):
        notional = getattr(order, "notional", None)
        avg_fill = getattr(order, "filled_avg_price", None)
        order_rows.append(
            {
                "Submitted": _text(getattr(order, "submitted_at", "")),
                "Symbol": _text(getattr(order, "symbol", "")),
                "Side": _text(getattr(order, "side", "")).upper(),
                "Type": _text(getattr(order, "order_type", "")),
                "Qty": _text(getattr(order, "qty", "")),
                "Notional": _money(_number(notional)) if notional is not None else "—",
                "Filled": _text(getattr(order, "filled_qty", "")),
                "Avg Fill": _money(_number(avg_fill)) if avg_fill is not None else "—",
                "Status": _text(getattr(order, "status", "")),
            }
        )

    return {
        "equity": _number(getattr(account, "equity", 0)),
        "cash": _number(getattr(account, "cash", 0)),
        "buying_power": _number(getattr(account, "buying_power", 0)),
        "positions": position_rows,
        "orders": order_rows,
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _render_account_tab(
    label: str,
    key_name: str,
    secret_name: str,
    slug: str,
) -> None:
    import pandas as pd
    import streamlit as st

    api_key = _secret(key_name)
    api_secret = _secret(secret_name)
    if not api_key or not api_secret:
        if label == "Trial #24":
            st.warning("Trial #24 keys not configured")
        else:
            st.warning(f"{label} keys not configured")
        return

    cache_key = f"account_snapshot_{slug}"
    cache = st.session_state.get(cache_key)
    if cache is None or time.time() - cache["loaded_at"] > 60:
        try:
            with st.spinner(f"Loading {label} account..."):
                snapshot = _fetch_account_snapshot(api_key, api_secret)
        except Exception:
            st.error(
                f"Could not load {label} account data. "
                "Check the credentials and API access."
            )
            return
        cache = {"loaded_at": time.time(), "snapshot": snapshot}
        st.session_state[cache_key] = cache

    if st.button("Refresh", key=f"refresh_{slug}"):
        st.session_state.pop(cache_key, None)
        st.rerun()

    snapshot = cache["snapshot"]
    metrics = st.columns(4)
    metrics[0].metric("Equity", _money(snapshot["equity"]))
    metrics[1].metric("Cash", _money(snapshot["cash"]))
    metrics[2].metric("Buying Power", _money(snapshot["buying_power"]))
    metrics[3].metric("Positions", len(snapshot["positions"]))
    st.caption(f"Paper account · updated {snapshot['as_of']}")

    st.markdown("**Positions**")
    if snapshot["positions"]:
        st.dataframe(pd.DataFrame(snapshot["positions"]), use_container_width=True)
    else:
        st.info("No open positions.")

    st.markdown("**Recent orders**")
    if snapshot["orders"]:
        st.dataframe(pd.DataFrame(snapshot["orders"]), use_container_width=True)
    else:
        st.info("No recent orders.")


def _render_trade_journal() -> None:
    import pandas as pd
    import streamlit as st

    st.markdown("**Trade journal**")
    default_path = Path(__file__).resolve().parent / "journal" / "trades.jsonl"
    journal_path = Path(os.environ.get("TRADER_JOURNAL", default_path))
    try:
        from trader.journal import load, summarize

        events = load(journal_path)
        stats = summarize(events)
    except Exception as exc:
        st.error(f"Could not read the trade journal: {exc}")
        return

    if not events:
        st.info(f"No journal entries found at {journal_path}.")
        return

    metrics = st.columns(3)
    metrics[0].metric("Closed trades", stats["trades"])
    metrics[1].metric("Win rate", f"{stats['win_rate']:.1%}")
    metrics[2].metric("Avg return", f"{stats['avg_return']:+.2%}")
    if stats["small_sample"]:
        st.caption("Small sample: fewer than 30 closed trades.")
    st.dataframe(pd.DataFrame(list(reversed(events[-25:]))), use_container_width=True)
    st.caption(f"Recent 25 events from {journal_path}")


def _render_gex_section() -> None:
    import pandas as pd
    import streamlit as st

    from market_tools.gex_calculator import (
        DEFAULT_HISTORY_PATH,
        STATUS_OK,
        collect_gex_snapshot,
        gamma_regime,
    )

    st.subheader("SPY Gamma Exposure (GEX)")
    st.caption(
        "Read-only observational view of the nearest non-expired SPY option chain. "
        "A dated snapshot is written once per US Eastern day to "
        f"`{DEFAULT_HISTORY_PATH.parent.name}/{DEFAULT_HISTORY_PATH.name}`. "
        "**Sign assumption:** Net GEX = Call GEX − Put GEX assumes dealers are "
        "net long calls (stabilizing positive gamma) and net short puts "
        "(amplifying negative gamma), the standard index-put-hedging convention. "
        "Open interest alone does not reveal true dealer direction."
    )

    cache = st.session_state.get("gex_snapshot_cache")
    if cache is None or time.time() - cache["loaded_at"] > 300:
        with st.spinner("Fetching SPY spot and options chain..."):
            snapshot = collect_gex_snapshot()
        cache = {"loaded_at": time.time(), "snapshot": snapshot}
        st.session_state["gex_snapshot_cache"] = cache

    snapshot = cache["snapshot"]

    if st.button("Refresh GEX snapshot"):
        st.session_state.pop("gex_snapshot_cache", None)
        st.rerun()

    if snapshot.status != STATUS_OK:
        st.error(f"GEX fetch failed: {snapshot.error_message}")
        st.caption(
            f"Today's status is recorded in `{DEFAULT_HISTORY_PATH.parent.name}/"
            f"{DEFAULT_HISTORY_PATH.name}`; FETCH_ERROR rows are never silently skipped."
        )
        st.warning(
            "**Disclaimer:** This GEX view is an observational data-collection tool, "
            "not an automated trading signal or investment advice."
        )
        return

    spot = snapshot.spot
    call_wall = snapshot.call_wall
    put_wall = snapshot.put_wall
    gamma_flip = snapshot.gamma_flip
    total = snapshot.net_gex_total
    regime = gamma_regime(total)

    metrics = st.columns(5)
    metrics[0].metric("SPY Spot", f"${spot:,.2f}" if spot is not None else "—")
    metrics[1].metric("Call Wall", f"${call_wall:,.2f}" if call_wall is not None else "—")
    metrics[2].metric("Put Wall", f"${put_wall:,.2f}" if put_wall is not None else "—")
    metrics[3].metric(
        "Gamma Flip",
        f"${gamma_flip:,.2f}" if gamma_flip is not None else "None ±10%",
    )
    metrics[4].metric("Total Net GEX", _format_gex_value(total))

    detail = st.columns(4)
    detail[0].metric(
        "0DTE Net GEX",
        _format_gex_value(snapshot.zero_dte_net_gex),
    )
    detail[1].metric(
        "Put/Call OI",
        f"{snapshot.put_call_oi_ratio:.2f}"
        if snapshot.put_call_oi_ratio is not None
        else "—",
    )
    detail[2].metric(
        "Max Pain",
        f"${snapshot.max_pain:,.2f}" if snapshot.max_pain is not None else "—",
    )
    detail[3].metric(
        "Gamma Flip Dist",
        f"${snapshot.dealer_gamma_flip_dist:,.2f}"
        if snapshot.dealer_gamma_flip_dist is not None
        else "—",
    )

    if total is not None and total >= 0:
        st.success(f"Net GEX regime: {regime} — dealers dampen moves.")
    else:
        st.warning(f"Net GEX regime: {regime} — dealers amplify moves.")

    st.caption(
        f"Expiration {snapshot.expiration} · Snapshot {snapshot.timestamp_et} · "
        f"Risk-free rate {snapshot.risk_free_rate:.3%}"
    )

    chart = pd.DataFrame(
        [{"strike": row.strike, "net_gex": row.net_gex} for row in snapshot.strikes]
    ).set_index("strike")
    st.markdown("**Strike-by-strike Net GEX** (current spot, fixed OI and IV)")
    st.bar_chart(chart, y="net_gex", use_container_width=True)

    st.warning(
        "**Disclaimer:** This GEX view is a read-only observational data-collection tool. "
        "It is not an automated trading signal, recommendation, or investment advice. "
        "Rows are logged once per US Eastern day, and FETCH_ERROR rows are recorded "
        "explicitly rather than silently skipped."
    )


def _render_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Paper Trading Dashboard", layout="wide")
    st.title("Paper Trading Dashboard")
    st.caption("Read-only account views for both paper-trading accounts.")

    with st.sidebar:
        st.header("Market tools")
        show_market_tools = st.checkbox("Show market viewer and GEX", value=False)

    account_tabs = st.tabs(["Discipline Bot", "Trial #24"])
    with account_tabs[0]:
        _render_account_tab(
            "Discipline Bot",
            "ALPACA_API_KEY",
            "ALPACA_API_SECRET",
            "discipline",
        )
        st.divider()
        _render_trade_journal()
    with account_tabs[1]:
        _render_account_tab(
            "Trial #24",
            "TRIAL24_ALPACA_API_KEY",
            "TRIAL24_ALPACA_SECRET_KEY",
            "trial24",
        )

    if show_market_tools:
        from market_viewer import render_html

        st.subheader("Market Viewer")
        st.components.v1.html(
            render_html(_payload()),
            height=1400,
            scrolling=True,
        )
        _render_gex_section()


def main() -> int:
    from market_viewer import main as viewer_main

    return viewer_main()


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        _render_streamlit()
    else:
        raise SystemExit(main())
