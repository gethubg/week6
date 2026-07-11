"""Custom MCP server wrapping yfinance (Tech Spec §4).

Transport: stdio (simplest for a local LangGraph client; swap to SSE if
deployed). Run directly with:

    uv run python -m mcp_server.server

The LangGraph Analysis Agent (agents/analysis_agent.py) talks to this
process as an MCP client; for tests it talks to price_provider.FakePriceProvider
directly instead, bypassing the MCP transport entirely.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.price_provider import YFinancePriceProvider, padded_window
from mcp_server.tools import compute_local_extremes

mcp = FastMCP("trade-coach-prices")
_provider = YFinancePriceProvider()


@mcp.tool()
def get_price_history(ticker: str, start: str, end: str) -> list[dict] | dict:
    """Daily OHLCV bars for `ticker` between `start` and `end` (ISO dates).
    Returns {"error": ...} instead of raising if the ticker/range has no data."""
    return _provider.get_price_history(ticker, start, end)


@mcp.tool()
def get_local_extremes(ticker: str, date: str, window_days: int = 10) -> dict:
    """Local low/high for `ticker` over a +/- window_days window around `date`.
    Returns {"error": ...} if there isn't enough price data to compute one."""
    start, end = padded_window(date, window_days)
    bars = _provider.get_price_history(ticker, start, end)
    if isinstance(bars, dict) and "error" in bars:
        return bars
    return compute_local_extremes(bars, date, window_days)


if __name__ == "__main__":
    mcp.run()
