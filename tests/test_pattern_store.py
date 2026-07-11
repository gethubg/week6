from agents.schemas import PatternClaim
from db.db import fetch_patterns, get_connection, init_db
from memory.embeddings import HashEmbedder
from memory.pattern_store import reconcile_patterns
from memory.vector_store import InMemoryVectorStore


def _setup():
    return InMemoryVectorStore(), HashEmbedder(), get_connection(":memory:")


def test_new_pattern_is_created_when_no_semantic_match():
    store, embedder, conn = _setup()
    init_db(conn)

    claim = PatternClaim(description="tends to sell winners early", evidence_trade_ids=["t1", "t2"], cited_concept="disposition effect")

    reconciled = reconcile_patterns(store, embedder, conn, [claim], total_closed_trades=4, now="2024-01-01")

    assert len(reconciled) == 1
    pattern = reconciled[0]
    assert pattern.description == "tends to sell winners early"
    assert pattern.evidence_trade_ids == ["t1", "t2"]
    assert pattern.confidence == 0.5  # 2 evidence trades / 4 total closed


def test_identical_pattern_on_a_later_run_merges_evidence_and_recomputes_confidence():
    store, embedder, conn = _setup()
    init_db(conn)

    first = PatternClaim(description="tends to sell winners early", evidence_trade_ids=["t1", "t2"], cited_concept="disposition effect")
    reconcile_patterns(store, embedder, conn, [first], total_closed_trades=4, now="2024-01-01")

    second = PatternClaim(description="tends to sell winners early", evidence_trade_ids=["t2", "t3"], cited_concept="disposition effect")
    reconciled = reconcile_patterns(store, embedder, conn, [second], total_closed_trades=6, now="2024-02-01")

    assert len(reconciled) == 1
    pattern = reconciled[0]
    # t2 was already evidence -- deduped, not double-counted.
    assert pattern.evidence_trade_ids == ["t1", "t2", "t3"]
    assert pattern.confidence == 0.5  # 3 / 6
    assert pattern.first_seen == "2024-01-01"
    assert pattern.last_updated == "2024-02-01"

    # Same pattern id persisted across both runs (a merge, not a duplicate).
    all_patterns = fetch_patterns(conn)
    assert len(all_patterns) == 1


def test_unrelated_pattern_description_creates_a_second_row():
    store, embedder, conn = _setup()
    init_db(conn)

    first = PatternClaim(description="tends to sell winners early", evidence_trade_ids=["t1"], cited_concept="disposition effect")
    reconcile_patterns(store, embedder, conn, [first], total_closed_trades=2, now="2024-01-01")

    second = PatternClaim(description="trades far more often on Mondays than any other day", evidence_trade_ids=["t2"], cited_concept="overtrading")
    reconcile_patterns(store, embedder, conn, [second], total_closed_trades=2, now="2024-01-15")

    all_patterns = fetch_patterns(conn)
    assert len(all_patterns) == 2


def test_confidence_is_capped_at_one():
    store, embedder, conn = _setup()
    init_db(conn)

    claim = PatternClaim(description="overtrades on earnings days", evidence_trade_ids=["t1", "t2", "t3"], cited_concept="overtrading")

    reconciled = reconcile_patterns(store, embedder, conn, [claim], total_closed_trades=2, now="2024-01-01")

    assert reconciled[0].confidence == 1.0
