"""Thin MCP client used by the Analysis Agent to reach mcp_server/server.py
over stdio (Tech Spec §4/§6.2).

This is the "real" path -- it spawns the server as a subprocess and talks
MCP over stdio. It requires network access (yfinance -> Yahoo Finance) and
is intentionally not exercised by the Phase 2 unit tests: agents/analysis_agent.py
takes an injectable ``extremes_fn`` so tests can run fully offline against
mcp_server.price_provider.FakePriceProvider instead. Wire this module in as
the default ``extremes_fn`` when running against live data.

get_local_extremes() keeps one subprocess + ClientSession alive across calls
in this process (see _ensure_persistent_session) instead of spawning a fresh
subprocess per call. score_trades() calls this once per trade leg, and
process-start overhead (~1s: interpreter boot + yfinance import + MCP stdio
handshake) was dominating analysis_node's runtime on any multi-leg upload.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])


@asynccontextmanager
async def _session():
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _call(session: ClientSession, ticker: str, date: str, window_days: int) -> dict:
    result = await session.call_tool(
        "get_local_extremes", {"ticker": ticker, "date": date, "window_days": window_days}
    )
    if result.structuredContent:
        return result.structuredContent
    # FastMCP only populates structuredContent when the tool's return
    # annotation is a specific schema (TypedDict/Pydantic/etc.); these
    # tools return a bare `dict`, so the actual payload arrives as a
    # JSON text block instead.
    for block in result.content:
        if block.type == "text":
            return json.loads(block.text)
    return {"error": "empty MCP response"}


async def get_local_extremes_async(ticker: str, date: str, window_days: int = 10) -> dict:
    """One-shot path: opens and closes its own subprocess/session. For
    anything already running inside an event loop. The sync
    get_local_extremes() below uses the persistent session instead."""
    async with _session() as session:
        return await _call(session, ticker, date, window_days)


_state: dict = {"loop": None, "session_cm": None, "session": None}


def _ensure_persistent_session() -> tuple[asyncio.AbstractEventLoop, ClientSession]:
    if _state["session"] is not None:
        return _state["loop"], _state["session"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    session_cm = _session()
    session = loop.run_until_complete(session_cm.__aenter__())
    _state.update(loop=loop, session_cm=session_cm, session=session)
    atexit.register(_close_persistent_session)
    return loop, session


def _close_persistent_session() -> None:
    loop, session_cm = _state["loop"], _state["session_cm"]
    if loop is None or session_cm is None:
        return
    try:
        loop.run_until_complete(session_cm.__aexit__(None, None, None))
    except Exception:
        pass  # best-effort cleanup at interpreter shutdown -- the subprocess
        # is killed with the parent process either way.


def get_local_extremes(ticker: str, date: str, window_days: int = 10) -> dict:
    """Sync wrapper -- convenient for LangGraph nodes that run synchronously.
    Reuses one persistent subprocess/session across calls in this process
    (see module docstring) instead of paying process-start cost per call."""
    loop, session = _ensure_persistent_session()
    return loop.run_until_complete(_call(session, ticker, date, window_days))
