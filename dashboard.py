"""Read-only dashboard: paper account, positions, orders, and the trade journal.

Keys come from Streamlit secrets when hosted, or from environment variables when run locally.
Nothing here can place orders. Keys are never written to the page or to the repo.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from trader.journal import load, summarize

st.set_page_config(page_title="Paper Trader", layout="wide")
st.title("Paper Trader")


def _secret(name: str) -> str:
    """Streamlit secrets first (hosted), then environment variables (local)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # no secrets file on this machine
    return os.getenv(name, "")


key = _secret("ALPACA_API_KEY") or os.getenv("ALPACA_KEY", "")
secret = _secret("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET", "")
if not key or not secret:
    st.error("Add ALPACA_API_KEY and ALPACA_API_SECRET in Streamlit secrets (hosted) or your environment (local).")
    st.stop()

client = TradingClient(key, secret, paper=True)

try:
    account = client.get_account()
    positions = client.get_all_positions()
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=25))
except Exception as exc:  # show the failure, do not crash the page
    st.error(f"Could not reach Alpaca: {exc}")
    st.stop()

equity = float(account.equity)
last_equity = float(account.last_equity)
c1, c2, c3 = st.columns(3)
c1.metric("Equity", f"${equity:,.2f}", f"{equity - last_equity:+,.2f} today")
c2.metric("Buying power", f"${float(account.buying_power):,.2f}")
c3.metric("Open positions", len(positions))

st.subheader("Positions")
if positions:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Symbol": p.symbol,
                    "Qty": float(p.qty),
                    "Market value": float(p.market_value),
                    "Unrealised P&L": float(p.unrealized_pl),
                }
                for p in positions
            ]
        ),
        hide_index=True,
    )
else:
    st.info("No open positions.")

st.subheader("Recent orders")
if orders:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Submitted": o.submitted_at.strftime("%Y-%m-%d %H:%M"),
                    "Symbol": o.symbol,
                    "Side": o.side.value.upper(),
                    "Type": o.order_type.value.upper() if o.order_type else "",
                    "Status": o.status.value.upper(),
                    "Client ID": o.client_order_id,
                }
                for o in orders
            ]
        ),
        hide_index=True,
    )
else:
    st.info("No orders yet.")

st.subheader("Trade journal")
journal_path = Path(os.getenv("TRADER_JOURNAL", "journal/trades.jsonl"))
events = load(journal_path)
if not events:
    st.info(
        f"No journal at {journal_path}. It fills in once the bot writes one "
        "(see TRADER_JOURNAL in the README)."
    )
else:
    stats = summarize(events)
    j1, j2, j3 = st.columns(3)
    j1.metric("Closed trades", stats["trades"])
    j2.metric("Win rate", f"{stats['win_rate']:.0%}")
    j3.metric("Skips", stats["skips"])
    if stats["small_sample"]:
        st.warning("Fewer than 30 closed trades. Treat these numbers as noise.")
    st.write("Exit reasons:", stats["exit_reasons"] or "none yet")
    st.dataframe(pd.DataFrame(events[-50:]), hide_index=True)
