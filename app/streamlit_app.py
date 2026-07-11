"""Streamlit UX (Tech Spec §12 / Product Spec §11):

1. Upload a Robinhood CSV export.
2. Live agent progress (ingestion -> analysis -> retrieval -> coach -> critic).
3. Trade-by-trade scorecard (timing percentile, verdict, linked news).
4. Pattern summary panel (behavioral tendencies, confidence, evidence).
5. Re-upload updates the persistent profile rather than starting fresh --
   the same SQLite file (and vector stores, once Pinecone is wired in) is
   reused across runs in this session.

Presentation (colors, badges, CSS) lives in app/ui.py -- see its module
docstring for the rule on what may vs. may not be interpolated into raw
HTML here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Makes `app.*` importable regardless of how this file is launched (a plain
# `python app/streamlit_app.py` / IDE run button only puts this file's own
# directory on sys.path, not the project root that `streamlit run` resolves to).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import ui
from app.pipeline import run_pipeline
from app.view_model import build_pattern_cards, build_scorecard_rows
from db.db import get_connection, init_db
from mcp_server.ticker_feed import fetch_latest_quotes

# Shown in the tape before any CSV is uploaded, or if an uploaded trade's
# ticker fails to quote -- keeps the tape non-empty on first load.
DEFAULT_WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]


# Matches the fragment's run_every below, so each auto-refresh tick pulls a
# genuinely fresh quote instead of serving a stale cached value.
_QUOTE_REFRESH_SECONDS = 30


@st.cache_data(ttl=_QUOTE_REFRESH_SECONDS, show_spinner=False)
def _cached_quotes(tickers: tuple[str, ...]) -> list:
    return fetch_latest_quotes(list(tickers))


@st.fragment(run_every=_QUOTE_REFRESH_SECONDS)
def render_ticker_tape() -> None:
    """Isolated as a fragment so only this section auto-reruns on a timer --
    the rest of the page (uploaded CSV, scorecard, expanders) stays put
    instead of resetting every 30s along with it."""
    watchlist_tickers = sorted({t.ticker for t in st.session_state.final_state.get("trades", [])}) or DEFAULT_WATCHLIST
    quotes = _cached_quotes(tuple(watchlist_tickers))
    tape_html = ui.ticker_tape_html(quotes)
    if tape_html:
        st.markdown(tape_html, unsafe_allow_html=True)


st.set_page_config(page_title="Trade Coach", layout="wide", page_icon="📈")
st.logo("assets/logo.svg")
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="tc-hero"><h1>Trade Coach</h1>'
    "<p>Upload a Robinhood CSV export to score your entry/exit timing and surface behavioral patterns.</p></div>",
    unsafe_allow_html=True,
)

if "conn" not in st.session_state:
    conn = get_connection(os.environ.get("SQLITE_PATH", "./db/trade_coach.db"))
    init_db(conn)
    st.session_state.conn = conn

if "final_state" not in st.session_state:
    st.session_state.final_state: dict = {}

render_ticker_tape()

with st.container(border=True):
    st.markdown("📤 **Upload your trades**")
    uploaded_file = st.file_uploader("Robinhood CSV export", type="csv", label_visibility="collapsed")
    run_clicked = st.button("Run Trade Coach", type="primary", disabled=uploaded_file is None)

if run_clicked and uploaded_file is not None:
    csv_text = uploaded_file.getvalue().decode("utf-8")
    progress = st.status("Running Trade Coach...", expanded=True)
    state: dict = {}
    for step in run_pipeline(csv_text, st.session_state.conn):
        node_name, update = next(iter(step.items()))
        state.update(update or {})
        progress.write(f"Ran `{node_name}`")
    progress.update(label="Done", state="complete")
    st.session_state.final_state = state

final_state = st.session_state.final_state

if not final_state:
    st.info("Upload a CSV and click Run Trade Coach to get started.")
else:
    timing_scores = final_state.get("timing_scores", [])
    total_legs = len(timing_scores)
    strong_pct = 100 * sum(1 for s in timing_scores if s.verdict == "strong") / total_legs if total_legs else 0
    weak_pct = 100 * sum(1 for s in timing_scores if s.verdict == "weak") / total_legs if total_legs else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Trades", len(final_state.get("trades", [])))
    metric_cols[1].metric("Legs Scored", total_legs)
    metric_cols[2].metric("Strong Timing", f"{strong_pct:.0f}%")
    metric_cols[3].metric("Weak Timing", f"{weak_pct:.0f}%")

    tab_scorecard, tab_patterns, tab_errors = st.tabs(["Scorecard", "Patterns", "Errors"])

    with tab_scorecard:
        rows = build_scorecard_rows(final_state)
        if not rows:
            st.info("No scored legs yet.")
        else:
            trades_seen: dict[str, list[dict]] = {}
            for row in rows:
                trades_seen.setdefault(row["trade_id"], []).append(row)

            for trade_rows in trades_seen.values():
                status = trade_rows[0]["status"]
                ticker = ui.escaped_ticker(trade_rows[0]["ticker"])
                leg_rows_html = "".join(
                    '<div class="tc-leg-row">'
                    f'{ui.badge(row["leg_type"].upper(), row["leg_type"])}'
                    f'<span class="tc-leg-date">{row["date"]}</span>'
                    f'<span class="tc-leg-price">${row["price"]:.2f}</span>'
                    f'<span class="tc-leg-pct">{ui.percentile_label(row["percentile"])}</span>'
                    f'{ui.badge((row["timing_verdict"] or "neutral").upper(), row["timing_verdict"] or "neutral")}'
                    "</div>"
                    for row in trade_rows
                )
                st.markdown(
                    f'<div class="tc-trade-card tc-trade-{status}">'
                    '<div class="tc-trade-header">'
                    f'<span class="tc-ticker">{ticker}</span>{ui.badge(status.upper(), status)}'
                    "</div>"
                    f"{leg_rows_html}"
                    "</div>",
                    unsafe_allow_html=True,
                )

                coach_verdict = trade_rows[0]["coach_verdict"]
                if coach_verdict:
                    st.caption(f"💬 {coach_verdict}")
                for row in trade_rows:
                    if row["news"]:
                        with st.expander(f'📰 News near {row["date"]} ({row["ticker"]} {row["leg_type"]})'):
                            st.write(row["news"])
                st.write("")

    with tab_patterns:
        cards = build_pattern_cards(final_state)
        if not cards:
            st.info("No behavioral patterns observed yet.")
        for card in cards:
            with st.container(border=True):
                st.markdown(ui.confidence_badge(card["confidence"]), unsafe_allow_html=True)
                st.markdown(f"**{card['description']}**")
                st.progress(card["confidence"], text=f"confidence: {card['confidence']:.0%}")
                st.caption(f"evidence trades: {', '.join(card['evidence_trade_ids'])} · last updated {card['last_updated']}")

    with tab_errors:
        errors = final_state.get("errors", [])
        if not errors:
            st.success("No errors this run.")
        for error in errors:
            st.warning(error)
