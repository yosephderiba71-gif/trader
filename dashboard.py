"""Compatibility entry point for the unified read-only market viewer.

The canonical implementation is ``market_viewer.py``. This file exists so the
old command still works:

    streamlit run dashboard.py   # Market Viewer + read-only SPY GEX tab
    python dashboard.py          # starts the local read-only viewer

No API keys are read, stored, or rendered. No broker or order endpoints are
imported. The SPY GEX tab fetches a public options chain and records a daily
observational snapshot in ``data/gex_history.csv``.
"""
from __future__ import annotations

import sys
import time


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

    from market_viewer import render_html

    st.set_page_config(page_title="Market Viewer", layout="wide")
    viewer_tab, gex_tab = st.tabs(["Market Viewer", "SPY Gamma Exposure (GEX)"])
    with viewer_tab:
        st.components.v1.html(render_html(_payload()), height=1400, scrolling=True)
    with gex_tab:
        _render_gex_section()


def main() -> int:
    from market_viewer import main as viewer_main

    return viewer_main()


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        _render_streamlit()
    else:
        raise SystemExit(main())
