"""Shared LangGraph state (Tech Spec §5.1)."""

from __future__ import annotations

from typing import TypedDict

from agents.schemas import CoachOutput, CriticOutput
from db.models import Leg, NewsContext, Pattern, Trade, TimingScore


class TradeCoachState(TypedDict, total=False):
    csv_raw: str
    trades: list[Trade]
    legs: list[Leg]
    timing_scores: list[TimingScore]
    news_context: list[NewsContext]
    news_by_leg: dict[str, list[NewsContext]]
    retrieved_concepts: list[str]
    patterns_retrieved: list[Pattern]
    coach_narrative: CoachOutput | None
    critic_verdict: CriticOutput | None
    errors: list[str]
    retry_count: int
