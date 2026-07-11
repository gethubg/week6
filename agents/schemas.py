"""Structured I/O contracts for the Coach and Critic agents (Tech Spec §6.3/§6.4)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TradeVerdict(BaseModel):
    trade_id: str
    verdict_text: str
    cited_score_ids: list[str] = Field(default_factory=list)  # leg_ids of cited TimingScores


class PatternClaim(BaseModel):
    description: str
    evidence_trade_ids: list[str]
    cited_concept: str


class CoachOutput(BaseModel):
    trade_verdicts: list[TradeVerdict] = Field(default_factory=list)
    patterns: list[PatternClaim] = Field(default_factory=list)


class RejectedClaim(BaseModel):
    claim_id: str
    reason: str


class CriticOutput(BaseModel):
    approved: list[str] = Field(default_factory=list)
    rejected: list[RejectedClaim] = Field(default_factory=list)
