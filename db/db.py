"""SQLite connection + read/write helpers for the trade-coach schema."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from db.models import Leg, NewsContext, Pattern, Trade, TimingScore

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(sqlite_path: str | None = None) -> sqlite3.Connection:
    path = sqlite_path or os.environ.get("SQLITE_PATH", ":memory:")
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # LangGraph's SqliteSaver checkpointer writes from a background executor
    # thread even when used synchronously, so the connection must allow
    # cross-thread use. Safe here because LangGraph serializes access to the
    # checkpointer (writes are awaited sequentially, never concurrent).
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()


# -- writes ------------------------------------------------------------------


def insert_trade(conn: sqlite3.Connection, trade: Trade) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO trade
            (id, ticker, open_date, close_date, status, quantity,
             avg_entry_price, avg_exit_price, realized_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade.id,
            trade.ticker,
            trade.open_date,
            trade.close_date,
            trade.status,
            trade.quantity,
            trade.avg_entry_price,
            trade.avg_exit_price,
            trade.realized_pnl,
        ),
    )


def insert_leg(conn: sqlite3.Connection, leg: Leg) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO leg (id, trade_id, type, date, price, quantity)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (leg.id, leg.trade_id, leg.type, leg.date, leg.price, leg.quantity),
    )


def insert_timing_score(conn: sqlite3.Connection, score: TimingScore) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO timing_score
            (leg_id, window_days, local_low, local_high, percentile, verdict)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            score.leg_id,
            score.window_days,
            score.local_low,
            score.local_high,
            score.percentile,
            score.verdict,
        ),
    )


def upsert_pattern(conn: sqlite3.Connection, pattern: Pattern) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO pattern
            (id, description, evidence_trade_ids, confidence, first_seen, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            pattern.id,
            pattern.description,
            json.dumps(pattern.evidence_trade_ids),
            pattern.confidence,
            pattern.first_seen,
            pattern.last_updated,
        ),
    )


def insert_news_context(conn: sqlite3.Connection, news: NewsContext) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO news_context
            (id, ticker, date, source, summary, pinecone_vector_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (news.id, news.ticker, news.date, news.source, news.summary, news.pinecone_vector_id),
    )


# -- reads --------------------------------------------------------------------


def fetch_trades(conn: sqlite3.Connection) -> list[Trade]:
    rows = conn.execute("SELECT * FROM trade ORDER BY open_date").fetchall()
    trades = [
        Trade(
            id=r["id"],
            ticker=r["ticker"],
            open_date=r["open_date"],
            close_date=r["close_date"],
            status=r["status"],
            quantity=r["quantity"],
            avg_entry_price=r["avg_entry_price"],
            avg_exit_price=r["avg_exit_price"],
            realized_pnl=r["realized_pnl"],
        )
        for r in rows
    ]
    for trade in trades:
        trade.legs = fetch_legs_for_trade(conn, trade.id)
    return trades


def fetch_legs_for_trade(conn: sqlite3.Connection, trade_id: str) -> list[Leg]:
    rows = conn.execute(
        "SELECT * FROM leg WHERE trade_id = ? ORDER BY date", (trade_id,)
    ).fetchall()
    return [
        Leg(
            id=r["id"],
            trade_id=r["trade_id"],
            type=r["type"],
            date=r["date"],
            price=r["price"],
            quantity=r["quantity"],
        )
        for r in rows
    ]


def fetch_patterns(conn: sqlite3.Connection) -> list[Pattern]:
    rows = conn.execute("SELECT * FROM pattern").fetchall()
    return [
        Pattern(
            id=r["id"],
            description=r["description"],
            evidence_trade_ids=json.loads(r["evidence_trade_ids"]),
            confidence=r["confidence"],
            first_seen=r["first_seen"],
            last_updated=r["last_updated"],
        )
        for r in rows
    ]
