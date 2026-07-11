from agents.schemas import CoachOutput, PatternClaim
from db.db import fetch_patterns, get_connection, init_db, insert_leg, insert_trade
from db.models import Leg, NewsContext, Trade
from graph.nodes import make_nodes
from llm.client import FakeLLMClient
from memory.embeddings import HashEmbedder
from memory.vector_store import InMemoryVectorStore
from rag.concepts import load_concept_docs
from rag.news_index import index_news


def _seeded_trade() -> Trade:
    trade = Trade(id="t1", ticker="AAPL", open_date="2024-01-01", close_date="2024-01-11", status="closed", quantity=10)
    trade.legs = [
        Leg(id="l1", trade_id="t1", type="buy", date="2024-01-01", price=100.0, quantity=10),
        Leg(id="l2", trade_id="t1", type="sell", date="2024-01-11", price=120.0, quantity=10),
    ]
    return trade


def test_retrieval_node_populates_news_patterns_and_concepts():
    conn = get_connection(":memory:")
    init_db(conn)
    trade = _seeded_trade()
    insert_trade(conn, trade)
    for leg in trade.legs:
        insert_leg(conn, leg)

    news_store = InMemoryVectorStore()
    embedder = HashEmbedder()
    index_news(
        news_store,
        embedder,
        conn,
        NewsContext(id="n1", ticker="AAPL", date="2024-01-10", source="wire", summary="Apple earnings beat"),
    )
    # Outside the +/-3d window around either leg date -- should not surface.
    index_news(
        news_store,
        embedder,
        conn,
        NewsContext(id="n2", ticker="AAPL", date="2024-06-01", source="wire", summary="Unrelated later news"),
    )

    nodes = make_nodes(conn, FakeLLMClient(), news_store=news_store, embedder=embedder)
    result = nodes["retrieval_node"]({"trades": [trade]})

    assert [n.id for n in result["news_context"]] == ["n1"]
    assert [n.id for n in result["news_by_leg"]["l1"]] == []
    assert [n.id for n in result["news_by_leg"]["l2"]] == ["n1"]
    assert result["patterns_retrieved"] == []
    assert set(result["retrieved_concepts"]) == set(load_concept_docs().keys())


def test_memory_node_reconciles_pattern_claims_into_store_and_db():
    conn = get_connection(":memory:")
    init_db(conn)
    trade = _seeded_trade()
    insert_trade(conn, trade)
    for leg in trade.legs:
        insert_leg(conn, leg)

    patterns_store = InMemoryVectorStore()
    embedder = HashEmbedder()
    nodes = make_nodes(conn, FakeLLMClient(), patterns_store=patterns_store, embedder=embedder)

    coach_narrative = CoachOutput(
        patterns=[PatternClaim(description="tends to sell winners early", evidence_trade_ids=["t1"], cited_concept="Disposition Effect")]
    )
    result = nodes["memory_node"]({"trades": [trade], "coach_narrative": coach_narrative})

    assert len(result["patterns_retrieved"]) == 1
    pattern = result["patterns_retrieved"][0]
    assert pattern.description == "tends to sell winners early"
    assert pattern.evidence_trade_ids == ["t1"]
    assert pattern.confidence == 1.0  # 1 evidence trade / 1 closed trade

    persisted = fetch_patterns(conn)
    assert len(persisted) == 1
    assert persisted[0].id == pattern.id

    # And it's queryable from the vector store for the next run's merge check.
    hits = patterns_store.query("default_user", embedder.embed("tends to sell winners early"), top_k=1)
    assert hits[0]["id"] == pattern.id


def test_memory_node_is_a_noop_when_coach_made_no_pattern_claims():
    conn = get_connection(":memory:")
    init_db(conn)
    nodes = make_nodes(conn, FakeLLMClient())

    result = nodes["memory_node"]({"trades": [], "coach_narrative": CoachOutput()})

    assert result == {}
    assert fetch_patterns(conn) == []
