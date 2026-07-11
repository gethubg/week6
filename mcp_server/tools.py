"""Pure computation over already-fetched price bars (Tech Spec §4). Kept
separate from price_provider.py so the windowing/extremes math is testable
without touching yfinance or the network at all.
"""

from __future__ import annotations

from mcp_server.price_provider import PriceBar


def compute_local_extremes(bars: list[PriceBar], date: str, window_days: int) -> dict:
    """Given bars already fetched for a padded window around ``date``,
    return {local_low, local_high, low_date, high_date} -- or an
    ``{"error": ...}`` dict if there's nothing to compute over (Tech Spec §4
    error contract: never fabricate a value)."""
    if not bars:
        return {"error": f"insufficient price data around {date} (window_days={window_days})"}

    low_bar = min(bars, key=lambda b: b["low"])
    high_bar = max(bars, key=lambda b: b["high"])

    return {
        "local_low": low_bar["low"],
        "local_high": high_bar["high"],
        "low_date": low_bar["date"],
        "high_date": high_bar["date"],
    }
