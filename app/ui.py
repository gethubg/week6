"""Presentation helpers for streamlit_app.py: injected CSS plus small
enum-keyed badge/formatting functions.

Only ever interpolate our own closed-vocabulary values (leg type, timing
verdict, trade status, confidence tier) into the HTML badges below --
never news summaries, coach narrative text, or pattern descriptions, all of
which come from Finnhub/LLM output and must stay on the escaped
st.write/st.markdown (no unsafe_allow_html) path.
"""

from __future__ import annotations

import html

from mcp_server.ticker_feed import LatestQuote

CSS = """
<style>
.tc-hero {
    padding: 1.5rem 0 1.1rem 0;
    border-bottom: 4px solid transparent;
    border-image: linear-gradient(90deg, #0E6B57, #B98900, #C1502E) 1;
    margin-bottom: 1.4rem;
}
.tc-hero h1 {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
    color: #1B2430;
}
.tc-hero p { color: #5B6572; font-size: 1.02rem; margin: 0; }

.tc-badge {
    display: inline-block;
    padding: 0.16rem 0.6rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    white-space: nowrap;
}
.tc-badge-strong { background: #E1F0EA; color: #0E6B57; }
.tc-badge-neutral { background: #FBF1D8; color: #92670A; }
.tc-badge-weak { background: #FBE4DC; color: #A33E22; }
.tc-badge-buy { background: #E3EDF7; color: #2A5C8A; }
.tc-badge-sell { background: #F0E6F7; color: #6B3FA0; }
.tc-badge-open { background: #FBF1D8; color: #92670A; }
.tc-badge-closed { background: #E1F0EA; color: #0E6B57; }
.tc-badge-unmatched { background: #ECEAE5; color: #5B6572; }

.tc-trade-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.1rem 1.3rem 0.7rem 1.3rem;
    margin-bottom: 0.35rem;
    box-shadow: 0 1px 3px rgba(27, 36, 48, 0.08);
    border-left: 5px solid #0E6B57;
}
.tc-trade-card.tc-trade-open { border-left-color: #B98900; }
.tc-trade-card.tc-trade-unmatched { border-left-color: #9AA1AA; }
.tc-trade-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}
.tc-ticker { font-size: 1.15rem; font-weight: 800; color: #1B2430; }

.tc-leg-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #F0EBDD;
}
.tc-leg-row:last-child { border-bottom: none; }
.tc-leg-date { color: #5B6572; min-width: 6rem; }
.tc-leg-price { font-weight: 600; min-width: 5rem; }
.tc-leg-pct { color: #5B6572; min-width: 6rem; }

[data-testid="stMetricValue"] { color: #0E6B57; }

.tc-ticker-tape {
    overflow: hidden;
    white-space: nowrap;
    background: #1B2430;
    border-radius: 10px;
    padding: 0.55rem 0;
    margin-bottom: 1.2rem;
}
.tc-ticker-track {
    display: inline-flex;
    animation: tc-scroll 40s linear infinite;
}
@keyframes tc-scroll {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}
.tc-ticker-item {
    display: inline-flex;
    align-items: center;
    padding: 0 1.6rem;
    font-weight: 700;
    font-size: 0.92rem;
    color: #FAF7F0;
    flex-shrink: 0;
}
.tc-ticker-sym { color: #FAF7F0; margin-right: 0.45rem; letter-spacing: 0.02em; }
.tc-ticker-item.tc-up { color: #6FD9B7; }
.tc-ticker-item.tc-down { color: #F3A38A; }
</style>
"""

_VALID_BADGE_KINDS = {
    "strong",
    "neutral",
    "weak",
    "buy",
    "sell",
    "open",
    "closed",
    "unmatched",
}


def badge(label: str, kind: str) -> str:
    """`kind` must be one of the app's own closed-vocabulary values -- see
    _VALID_BADGE_KINDS. Rejects anything else so this can never become a
    path for rendering untrusted text as HTML."""
    if kind not in _VALID_BADGE_KINDS:
        raise ValueError(f"badge() kind must be one of {sorted(_VALID_BADGE_KINDS)}, got {kind!r}")
    return f'<span class="tc-badge tc-badge-{kind}">{label}</span>'


def percentile_label(percentile: float | None) -> str:
    return f"{percentile:.0f}th pct" if percentile is not None else "—"


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.7:
        return badge("HIGH CONFIDENCE", "strong")
    if confidence >= 0.4:
        return badge("MODERATE CONFIDENCE", "neutral")
    return badge("LOW CONFIDENCE", "weak")


def escaped_ticker(ticker: str) -> str:
    """Trade tickers come from the uploaded CSV's Instrument column --
    unlike verdict/status/leg-type, they are NOT a closed vocabulary, so
    anything rendering one via unsafe_allow_html must escape it first."""
    return html.escape(str(ticker))


def ticker_tape_html(quotes: list[LatestQuote]) -> str:
    """Builds a seamlessly-looping scroll track by rendering the sequence
    twice back to back and animating a translateX(-50%) -- at the halfway
    point the visible content is identical to the start, so the loop has no
    visible jump."""
    if not quotes:
        return ""

    items = []
    for quote in quotes:
        direction = "tc-up" if quote["change_pct"] >= 0 else "tc-down"
        arrow = "▲" if quote["change_pct"] >= 0 else "▼"
        items.append(
            f'<span class="tc-ticker-item {direction}">'
            f'<span class="tc-ticker-sym">{escaped_ticker(quote["ticker"])}</span>'
            f'${quote["price"]:,.2f} {arrow} {abs(quote["change_pct"]):.2f}%'
            "</span>"
        )
    sequence = "".join(items)
    return f'<div class="tc-ticker-tape"><div class="tc-ticker-track">{sequence}{sequence}</div></div>'
