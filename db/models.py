"""Dataclasses mirroring the schema in db/schema.sql (Product Spec §4)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Leg:
    id: str
    trade_id: str | None
    type: str  # 'buy' | 'sell' | 'dividend'
    date: str  # ISO date
    price: float
    quantity: float


@dataclass
class Trade:
    id: str
    ticker: str
    open_date: str
    status: str  # 'open' | 'closed' | 'unmatched'
    quantity: float
    close_date: str | None = None
    avg_entry_price: float | None = None
    avg_exit_price: float | None = None
    realized_pnl: float | None = None
    legs: list[Leg] = field(default_factory=list)


@dataclass
class TimingScore:
    leg_id: str
    window_days: int
    local_low: float | None
    local_high: float | None
    percentile: float | None
    verdict: str | None


@dataclass
class Pattern:
    id: str
    description: str
    evidence_trade_ids: list[str]
    confidence: float
    first_seen: str
    last_updated: str


@dataclass
class NewsContext:
    id: str
    ticker: str
    date: str
    source: str | None
    summary: str | None
    pinecone_vector_id: str | None = None
