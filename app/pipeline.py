"""Wires up real vs. offline dependencies for a run, based on which env vars
are set (Tech Spec §12 / .env.example). Kept separate from streamlit_app.py
so it's importable and testable without Streamlit itself.

Price data always goes through the real MCP/yfinance path (mcp_server.client
.get_local_extremes) -- yfinance needs network but no API key. If that
network call fails, score_trades already records the failure in
state.errors per leg (Tech Spec §4 error contract) instead of crashing, so
there's no separate "offline price" mode to wire here.
"""

from __future__ import annotations

import os
import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from graph.build_graph import build_graph
from graph.state import TradeCoachState
from llm.client import AnthropicLLMClient, FakeLLMClient, LLMClient, NebiusLLMClient
from mcp_server.client import get_local_extremes
from mcp_server.news_provider import FinnhubProvider, NewsProvider
from memory.embeddings import Embedder, HashEmbedder, NebiusEmbedder, OpenAIEmbedder
from memory.vector_store import InMemoryVectorStore, PineconeVectorStore, VectorStore

# Populates os.environ from a `.env` file in the current working directory
# (i.e. wherever `streamlit run` / pytest was launched from) before any of
# the make_* factories below check for API keys. No-ops if no .env is found,
# and never overrides a variable the shell already exported.
load_dotenv()


def make_llm_client() -> LLMClient:
    if os.environ.get("NEBIUS_API_KEY"):
        return NebiusLLMClient()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLMClient()
    return FakeLLMClient()


def make_news_provider() -> NewsProvider | None:
    """None means "don't fetch live news" -- retrieval_node then just uses
    whatever's already indexed (offline/test fixtures)."""
    if os.environ.get("NEWS_API_KEY"):
        return FinnhubProvider()
    return None


def make_embedder() -> Embedder:
    if os.environ.get("NEBIUS_API_KEY"):
        return NebiusEmbedder()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    return HashEmbedder()


def make_vector_store(index_env_var: str) -> VectorStore:
    if os.environ.get("PINECONE_API_KEY") and os.environ.get(index_env_var):
        return PineconeVectorStore(os.environ[index_env_var])
    return InMemoryVectorStore()


def make_checkpointer(conn: sqlite3.Connection) -> SqliteSaver:
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def run_pipeline(csv_text: str, conn: sqlite3.Connection, thread_id: str = "default"):
    """Builds the graph with whatever real/offline dependencies are
    available (see the make_* functions above) and streams node-by-node
    state updates so the caller can show live progress."""
    graph = build_graph(
        conn,
        make_llm_client(),
        extremes_fn=get_local_extremes,
        news_store=make_vector_store("PINECONE_INDEX_NEWS"),
        patterns_store=make_vector_store("PINECONE_INDEX_PATTERNS"),
        embedder=make_embedder(),
        checkpointer=make_checkpointer(conn),
        news_provider=make_news_provider(),
    )

    initial_state: TradeCoachState = {"csv_raw": csv_text, "errors": [], "retry_count": 0}
    yield from graph.stream(initial_state, config={"configurable": {"thread_id": thread_id}})
