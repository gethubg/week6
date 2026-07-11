"""Qualitative pattern memory (Product Spec §7 / Tech Spec §10).

Deterministic, inspectable merge logic -- no LLM-driven consolidation. On
each run, approved PatternClaims from the Coach/Critic are either merged
into a semantically-similar existing Pattern (evidence_trade_ids unioned,
confidence recomputed) or inserted as a new Pattern row.
"""

from __future__ import annotations

import sqlite3
import uuid

from agents.schemas import PatternClaim
from db.db import fetch_patterns, upsert_pattern
from db.models import Pattern
from memory.embeddings import Embedder
from memory.vector_store import VectorStore

# V0 has a single hardcoded user -- "namespace per user" collapses to one
# namespace until multi-user support exists (Tech Spec §13).
PATTERNS_NAMESPACE = "default_user"
MERGE_SIMILARITY_THRESHOLD = 0.8


def _confidence(evidence_count: int, total_closed_trades: int) -> float:
    if total_closed_trades <= 0:
        return 0.0
    return min(1.0, evidence_count / total_closed_trades)


def reconcile_patterns(
    store: VectorStore,
    embedder: Embedder,
    conn: sqlite3.Connection,
    candidate_claims: list[PatternClaim],
    total_closed_trades: int,
    now: str,
    threshold: float = MERGE_SIMILARITY_THRESHOLD,
) -> list[Pattern]:
    existing_patterns = {p.id: p for p in fetch_patterns(conn)}
    reconciled: list[Pattern] = []

    for claim in candidate_claims:
        query_embedding = embedder.embed(claim.description)
        matches = store.query(namespace=PATTERNS_NAMESPACE, query_embedding=query_embedding, top_k=1)

        if matches and matches[0]["score"] >= threshold and matches[0]["id"] in existing_patterns:
            pattern = existing_patterns[matches[0]["id"]]
            merged_evidence = sorted(set(pattern.evidence_trade_ids) | set(claim.evidence_trade_ids))
            pattern.evidence_trade_ids = merged_evidence
            pattern.confidence = _confidence(len(merged_evidence), total_closed_trades)
            pattern.last_updated = now
        else:
            deduped_evidence = list(dict.fromkeys(claim.evidence_trade_ids))
            pattern = Pattern(
                id=uuid.uuid4().hex,
                description=claim.description,
                evidence_trade_ids=deduped_evidence,
                confidence=_confidence(len(deduped_evidence), total_closed_trades),
                first_seen=now,
                last_updated=now,
            )
            store.upsert(
                namespace=PATTERNS_NAMESPACE,
                id=pattern.id,
                embedding=query_embedding,
                metadata={"description": pattern.description},
            )
            existing_patterns[pattern.id] = pattern

        upsert_pattern(conn, pattern)
        reconciled.append(pattern)

    conn.commit()
    return reconciled
