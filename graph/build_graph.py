"""Graph wiring (Tech Spec §5.2):

    START -> ingestion_node -> analysis_node -> retrieval_node -> coach_node -> critic_node
    critic_node -[rejected claims, retry_count < 1]-> bump_retry_node -> coach_node
    critic_node -[approved, or already retried once]-> finalize_node -> memory_node -> END

retrieval_node populates news_context (RAG), patterns_retrieved (existing
Pattern rows), and retrieved_concepts (the concept KB) ahead of the Coach
call. memory_node reconciles this run's approved pattern claims into the
Pattern table + patterns vector index after the Critic has had its say.

Checkpointing uses LangGraph's SQLite checkpointer so a Streamlit rerun can
resume a run rather than re-invoking ingestion from scratch.
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analysis_agent import ExtremesFn
from agents.ingestion_agent import AmbiguousRowClassifier, default_ambiguous_classifier
from graph.nodes import make_nodes, route_after_critic
from graph.state import TradeCoachState
from llm.client import LLMClient
from mcp_server.client import get_local_extremes
from mcp_server.news_provider import NewsProvider
from memory.embeddings import Embedder
from memory.vector_store import VectorStore
from tools.lot_matching import IdGenerator


def build_graph(
    conn: sqlite3.Connection,
    llm_client: LLMClient,
    extremes_fn: ExtremesFn = get_local_extremes,
    ambiguous_classifier: AmbiguousRowClassifier = default_ambiguous_classifier,
    id_gen: IdGenerator | None = None,
    news_store: VectorStore | None = None,
    patterns_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    checkpointer: SqliteSaver | None = None,
    news_provider: NewsProvider | None = None,
) -> CompiledStateGraph:
    nodes = make_nodes(
        conn,
        llm_client,
        extremes_fn,
        ambiguous_classifier,
        id_gen,
        news_store,
        patterns_store,
        embedder,
        news_provider,
    )

    graph = StateGraph(TradeCoachState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "ingestion_node")
    graph.add_edge("ingestion_node", "analysis_node")
    graph.add_edge("analysis_node", "retrieval_node")
    graph.add_edge("retrieval_node", "coach_node")
    graph.add_edge("coach_node", "critic_node")
    graph.add_conditional_edges(
        "critic_node", route_after_critic, {"retry": "bump_retry_node", "finalize": "finalize_node"}
    )
    graph.add_edge("bump_retry_node", "coach_node")
    graph.add_edge("finalize_node", "memory_node")
    graph.add_edge("memory_node", END)

    return graph.compile(checkpointer=checkpointer)
