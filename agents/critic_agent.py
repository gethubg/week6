"""Critic Agent (Product Spec §5.4 / Tech Spec §6.4).

A thin wrapper over tools/critic_verify.py -- deterministic and
LLM-free by design, since its whole job is to catch the Coach's LLM output
being ungrounded.
"""

from __future__ import annotations

from langsmith import traceable

from agents.schemas import CoachOutput, CriticOutput
from db.models import Trade, TimingScore
from tools.critic_verify import verify_claims


@traceable(name="critic_agent", run_type="chain")
def run_critic(
    coach_output: CoachOutput,
    trades: list[Trade],
    timing_scores: list[TimingScore],
    retrieved_concepts: list[str],
) -> CriticOutput:
    return verify_claims(coach_output, trades, timing_scores, retrieved_concepts)
