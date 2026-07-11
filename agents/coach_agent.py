"""Coach Agent (Product Spec §5.3 / Tech Spec §6.3).

Synthesizes TimingScores + retrieved news + retrieved behavioral concepts +
prior Pattern rows into narrative verdicts. The LLM call is the only
non-deterministic piece in the pipeline; everything it's grounded in
(scores, concepts, patterns) was computed or retrieved upstream.
"""

from __future__ import annotations

from langsmith import traceable

from db.models import NewsContext, Pattern, Trade, TimingScore
from llm.client import LLMClient
from agents.schemas import CoachOutput

SYSTEM_PROMPT = """\
You are a trading behavior coach. You are given computed timing scores for \
a trader's closed trades, retrieved news/earnings context, a small set of \
retrieved behavioral-finance concepts, and any previously observed patterns.

Rules you must follow:
- Every trade verdict must cite the leg-level timing score ids (cited_score_ids) \
it is based on.
- Every pattern claim must cite evidence_trade_ids that appear in the trades \
provided below, and a cited_concept that appears in the retrieved concepts \
list below -- never invent a concept that wasn't retrieved.
- If you cannot ground a claim in the provided data, do not make it.
"""


def _format_timing_scores(trades: list[Trade], timing_scores: list[TimingScore]) -> str:
    scores_by_leg_id = {s.leg_id: s for s in timing_scores}
    lines = []
    for trade in trades:
        lines.append(f"Trade {trade.id} ({trade.ticker}, status={trade.status}):")
        for leg in trade.legs:
            score = scores_by_leg_id.get(leg.id)
            if score is None:
                continue
            lines.append(
                f"  leg {leg.id} [{leg.type}] on {leg.date} @ {leg.price}: "
                f"percentile={score.percentile:.1f} verdict={score.verdict} "
                f"(window {score.window_days}d, range [{score.local_low}, {score.local_high}])"
            )
    return "\n".join(lines) if lines else "(no timing scores available)"


def build_coach_prompt(
    trades: list[Trade],
    timing_scores: list[TimingScore],
    news_context: list[NewsContext],
    retrieved_concepts: list[str],
    patterns_retrieved: list[Pattern],
    critic_feedback: list[str] | None = None,
) -> str:
    sections = [
        "## Timing scores",
        _format_timing_scores(trades, timing_scores),
        "\n## Retrieved news/earnings context",
        "\n".join(f"- {n.ticker} ({n.date}): {n.summary}" for n in news_context) or "(none)",
        "\n## Retrieved behavioral concepts (only cite these, verbatim)",
        "\n".join(f"- {c}" for c in retrieved_concepts) or "(none retrieved -- do not make pattern claims)",
        "\n## Previously observed patterns",
        "\n".join(f"- {p.description} (confidence={p.confidence})" for p in patterns_retrieved) or "(none yet)",
    ]
    if critic_feedback:
        sections.append("\n## Your previous attempt was rejected for:")
        sections.append("\n".join(f"- {reason}" for reason in critic_feedback))
        sections.append("Revise your claims so every one is grounded, or drop it.")
    return "\n".join(sections)


@traceable(name="coach_agent", run_type="chain")
def run_coach(
    trades: list[Trade],
    timing_scores: list[TimingScore],
    news_context: list[NewsContext],
    retrieved_concepts: list[str],
    patterns_retrieved: list[Pattern],
    llm_client: LLMClient,
    critic_feedback: list[str] | None = None,
) -> CoachOutput:
    prompt = build_coach_prompt(trades, timing_scores, news_context, retrieved_concepts, patterns_retrieved, critic_feedback)
    return llm_client.generate_structured(prompt, CoachOutput, system=SYSTEM_PROMPT)
