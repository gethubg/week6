"""LangGraph node functions (Tech Spec §5.2). Each node is a closure over
the run's dependencies (DB connection, LLM client, price-extremes function,
vector stores/embedder) so the graph itself stays a pure wiring diagram in
build_graph.py.

Concept-KB retrieval note: retrieve_concept() in rag/concepts.py does
similarity-threshold retrieval given a candidate pattern description -- but
the Coach doesn't have a candidate description until *after* it runs, so a
per-claim similarity lookup can't happen before the Coach's first call. With
only ~10 short docs in the KB (Tech Spec §9.2), there's no real cost to just
handing the Coach the full list of concept titles up front instead of a
similarity-gated shortlist; retrieve_concept() remains available standalone
for the groundedness eval (Phase 7) to double-check a citation was actually
the best KB match for what the Coach described, not just *a* valid one.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Callable

from agents.analysis_agent import ExtremesFn, score_trades
from agents.coach_agent import run_coach
from agents.critic_agent import run_critic
from agents.ingestion_agent import AmbiguousRowClassifier, default_ambiguous_classifier, run_ingestion
from agents.schemas import CoachOutput
from db.db import fetch_patterns
from db.models import NewsContext
from graph.state import TradeCoachState
from llm.client import LLMClient
from mcp_server.client import get_local_extremes
from mcp_server.news_provider import NewsProvider
from memory.embeddings import Embedder, HashEmbedder
from memory.pattern_store import reconcile_patterns
from memory.vector_store import InMemoryVectorStore, VectorStore
from rag.concepts import load_concept_docs
from rag.news_index import fetch_and_index_news, retrieve_news_context
from tools.lot_matching import IdGenerator

MAX_RETRIES = 1


def make_nodes(
    conn: sqlite3.Connection,
    llm_client: LLMClient,
    extremes_fn: ExtremesFn = get_local_extremes,
    ambiguous_classifier: AmbiguousRowClassifier = default_ambiguous_classifier,
    id_gen: IdGenerator | None = None,
    news_store: VectorStore | None = None,
    patterns_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    news_provider: NewsProvider | None = None,
) -> dict[str, Callable[[TradeCoachState], dict]]:
    news_store = news_store if news_store is not None else InMemoryVectorStore()
    patterns_store = patterns_store if patterns_store is not None else InMemoryVectorStore()
    embedder = embedder or HashEmbedder()

    def ingestion_node(state: TradeCoachState) -> dict:
        result = run_ingestion(state["csv_raw"], conn, ambiguous_classifier, id_gen=id_gen)
        errors = list(state.get("errors", []))
        errors += [f"skipped unclassified row: {row}" for row in result.skipped_rows]
        return {"trades": result.trades, "legs": result.legs, "errors": errors}

    def analysis_node(state: TradeCoachState) -> dict:
        scores, score_errors = score_trades(state["trades"], conn, extremes_fn)
        errors = list(state.get("errors", [])) + score_errors
        return {"timing_scores": scores, "errors": errors}

    def retrieval_node(state: TradeCoachState) -> dict:
        news_context: list[NewsContext] = []
        news_by_leg: dict[str, list[NewsContext]] = {}
        seen_ids: set[str] = set()
        fetched: set[tuple[str, str]] = set()
        for trade in state["trades"]:
            for leg in trade.legs:
                if leg.type not in ("buy", "sell"):
                    continue
                if news_provider is not None and (trade.ticker, leg.date) not in fetched:
                    fetched.add((trade.ticker, leg.date))
                    fetch_and_index_news(news_store, embedder, conn, news_provider, trade.ticker, leg.date)
                # retrieve_news_context is date-windowed (+/- window_days), so a
                # leg's linked news won't share its exact date -- news_by_leg
                # keeps this leg's own results instead of the flat list below,
                # which is deduped globally for the Coach's narrative context.
                leg_news = retrieve_news_context(news_store, embedder, trade.ticker, leg.date)
                news_by_leg[leg.id] = leg_news
                for news in leg_news:
                    if news.id not in seen_ids:
                        seen_ids.add(news.id)
                        news_context.append(news)

        return {
            "news_context": news_context,
            "news_by_leg": news_by_leg,
            "patterns_retrieved": fetch_patterns(conn),
            "retrieved_concepts": list(load_concept_docs().keys()),
        }

    def coach_node(state: TradeCoachState) -> dict:
        critic = state.get("critic_verdict")
        feedback = [r.reason for r in critic.rejected] if critic else None
        coach_output = run_coach(
            state["trades"],
            state["timing_scores"],
            state.get("news_context", []),
            state.get("retrieved_concepts", []),
            state.get("patterns_retrieved", []),
            llm_client,
            critic_feedback=feedback,
        )
        return {"coach_narrative": coach_output}

    def critic_node(state: TradeCoachState) -> dict:
        critic_output = run_critic(
            state["coach_narrative"],
            state["trades"],
            state["timing_scores"],
            state.get("retrieved_concepts", []),
        )
        return {"critic_verdict": critic_output}

    def bump_retry_node(state: TradeCoachState) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}

    def finalize_node(state: TradeCoachState) -> dict:
        critic = state["critic_verdict"]
        if not critic.rejected:
            return {}
        coach = state["coach_narrative"]
        rejected_ids = {r.claim_id for r in critic.rejected}
        stripped = CoachOutput(
            trade_verdicts=[v for v in coach.trade_verdicts if f"trade:{v.trade_id}" not in rejected_ids],
            patterns=[p for i, p in enumerate(coach.patterns) if f"pattern:{i}" not in rejected_ids],
        )
        return {"coach_narrative": stripped}

    def memory_node(state: TradeCoachState) -> dict:
        coach = state["coach_narrative"]
        if not coach.patterns:
            return {}
        total_closed = len([t for t in state["trades"] if t.status == "closed"])
        now = datetime.now(timezone.utc).date().isoformat()
        reconciled = reconcile_patterns(patterns_store, embedder, conn, coach.patterns, total_closed, now)
        return {"patterns_retrieved": reconciled}

    return {
        "ingestion_node": ingestion_node,
        "analysis_node": analysis_node,
        "retrieval_node": retrieval_node,
        "coach_node": coach_node,
        "critic_node": critic_node,
        "bump_retry_node": bump_retry_node,
        "finalize_node": finalize_node,
        "memory_node": memory_node,
    }


def route_after_critic(state: TradeCoachState) -> str:
    critic = state["critic_verdict"]
    if critic.rejected and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "finalize"
