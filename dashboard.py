"""Compatibility entry point for the unified read-only market viewer.

The canonical implementation is ``market_viewer.py``. This file exists so the
old command still works:

    streamlit run dashboard.py   # renders the same HTML inside Streamlit
    python dashboard.py          # starts the local read-only viewer

No API keys are read, stored, or rendered. No broker or order endpoints are
imported.
"""
from __future__ import annotations

import sys


def _payload() -> dict:
    from market_viewer import build_payload, default_provider

    return build_payload(default_provider())


def _render_streamlit() -> None:
    import streamlit as st

    from market_viewer import render_html

    st.set_page_config(page_title="Market Viewer", layout="wide")
    st.components.v1.html(render_html(_payload()), height=1400, scrolling=True)


def main() -> int:
    from market_viewer import main as viewer_main

    return viewer_main()


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        _render_streamlit()
    else:
        raise SystemExit(main())
