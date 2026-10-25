"""Read-only dashboard for the paper account.

Run locally:  streamlit run dashboard.py
Keys come from environment variables. Nothing is hard-coded, so nothing can leak into git.
"""
import os

import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

st.set_page_config(page_title="Paper Trader", layout="wide")
st.title("Paper Trader")

key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY")
secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET")
if not key or not secret:
    st.error("Set ALPACA_API_KEY and ALPACA_API_SECRET in your environment, then restart Streamlit.")
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
