"""News/earnings context RAG pipeline (Product Spec §6 / Tech Spec §9.1).

Retrieval at Coach time is scoped to a +/- window around a trade leg's date
so the Coach can distinguish "sold before earnings" from "sold with no
catalyst" -- not a general-purpose news search.
"""

from __future__ import annotations

import sqlite3

from db.db import insert_news_context
from db.models import NewsContext
from mcp_server.news_provider import NewsProvider
from mcp_server.price_provider import padded_window
from memory.embeddings import Embedder
from memory.vector_store import UpsertRecord, VectorStore

DEFAULT_TOP_K = 3
DEFAULT_DATE_WINDOW_DAYS = 3
# index_news does one embed + one upsert network call per article; a
# high-coverage ticker (e.g. AAPL) can return 200+ articles for even a
# single week from Finnhub, which turns one retrieval_node call into
# minutes of sequential network round-trips. retrieve_news_context only
# ever wants DEFAULT_TOP_K, so there's no value in indexing more than a
# small multiple of that.
MAX_ARTICLES_PER_FETCH = 15


def _date_to_ordinal(date: str) -> int:
    """Pinecone's $gte/$lte require a numeric metadata value -- an ISO date
    string sorts correctly in Python but the real API rejects it, so range
    filtering goes through this YYYYMMDD int instead. `date` (the ISO
    string) stays in metadata too, for display when reconstructing
    NewsContext from a query result."""
    return int(date.replace("-", ""))


def _metadata(news: NewsContext) -> dict:
    return {
        "ticker": news.ticker,
        "date": news.date,
        "date_ordinal": _date_to_ordinal(news.date),
        "source": news.source,
        "summary": news.summary,
    }


def index_news(store: VectorStore, embedder: Embedder, conn: sqlite3.Connection, news: NewsContext) -> None:
    news.pinecone_vector_id = news.pinecone_vector_id or news.id
    store.upsert(
        namespace=news.ticker,
        id=news.pinecone_vector_id,
        embedding=embedder.embed(news.summary or ""),
        metadata=_metadata(news),
    )
    insert_news_context(conn, news)


def index_news_batch(store: VectorStore, embedder: Embedder, conn: sqlite3.Connection, news_items: list[NewsContext]) -> None:
    """Same effect as calling index_news() once per item, but one embed_batch
    call and one upsert_batch call instead of one embed + one upsert per
    item -- the network-round-trip cost fetch_and_index_news was paying per
    article. All items must share a namespace (ticker), since upsert_batch
    is a single-namespace call."""
    if not news_items:
        return

    embeddings = embedder.embed_batch([news.summary or "" for news in news_items])
    records: list[UpsertRecord] = []
    for news, embedding in zip(news_items, embeddings):
        news.pinecone_vector_id = news.pinecone_vector_id or news.id
        records.append(UpsertRecord(id=news.pinecone_vector_id, embedding=embedding, metadata=_metadata(news)))

    store.upsert_batch(namespace=news_items[0].ticker, records=records)
    for news in news_items:
        insert_news_context(conn, news)


def retrieve_news_context(
    store: VectorStore,
    embedder: Embedder,
    ticker: str,
    trade_leg_date: str,
    window_days: int = DEFAULT_DATE_WINDOW_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> list[NewsContext]:
    start, end = padded_window(trade_leg_date, window_days)
    query_embedding = embedder.embed(f"{ticker} news and earnings around {trade_leg_date}")
    results = store.query(
        namespace=ticker,
        query_embedding=query_embedding,
        top_k=top_k,
        filter={"date_ordinal": {"$gte": _date_to_ordinal(start), "$lte": _date_to_ordinal(end)}},
    )
    return [
        NewsContext(
            id=r["id"],
            ticker=ticker,
            date=r["metadata"]["date"],
            source=r["metadata"].get("source"),
            summary=r["metadata"].get("summary"),
            pinecone_vector_id=r["id"],
        )
        for r in results
    ]


def fetch_and_index_news(
    store: VectorStore,
    embedder: Embedder,
    conn: sqlite3.Connection,
    provider: NewsProvider,
    ticker: str,
    trade_leg_date: str,
    window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> None:
    """Live path: pull articles for `ticker` in the +/- window around
    `trade_leg_date` from `provider` and index them, so a later
    retrieve_news_context call has something to find. Errors from the
    provider (missing key, network failure, no results) are swallowed --
    retrieval just falls back to whatever's already indexed."""
    start, end = padded_window(trade_leg_date, window_days)
    articles = provider.get_news(ticker, start, end)
    if isinstance(articles, dict):
        return

    news_items = [
        NewsContext(
            id=f"{ticker}:{article['url'] or article['date'] + article['title']}",
            ticker=ticker,
            date=article["date"],
            source=article["source"],
            summary=article["summary"],
        )
        for article in articles[:MAX_ARTICLES_PER_FETCH]
    ]
    index_news_batch(store, embedder, conn, news_items)
