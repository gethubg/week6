"""LLM-as-judge + Critic precision/recall eval layer (Tech Spec §11).

Two independent checks:

1. ``evaluate_critic_precision_recall`` -- runs tools.critic_verify against
   evals/golden_claims.jsonl, a hand-labeled set of intentionally grounded
   and intentionally ungrounded synthetic claims over a fixed reference
   Trade/TimingScore world (below). This is deterministic and catches
   regressions in the Critic's rejection logic itself.
2. ``judge_groundedness`` -- an LLM-as-judge over a *real* CoachOutput: does
   each claim's prose actually match what its cited scores show? This is
   the check the deterministic Critic can't do (it verifies citations
   exist, not that the narrative is a faithful description of them).

Run standalone: uv run python -m evals.groundedness_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.schemas import CoachOutput, PatternClaim, TradeVerdict
from db.models import Leg, Trade, TimingScore
from llm.client import LLMClient
from tools.critic_verify import verify_claims
from pydantic import BaseModel, Field

GOLDEN_CLAIMS_PATH = Path(__file__).parent / "golden_claims.jsonl"

# -- fixed reference world, shared by every case in golden_claims.jsonl -----


def _trade(id: str, ticker: str, buy_price: float, buy_date: str, sell_price: float, sell_date: str, buy_leg_id: str, sell_leg_id: str) -> Trade:
    trade = Trade(id=id, ticker=ticker, open_date=buy_date, close_date=sell_date, status="closed", quantity=10)
    trade.legs = [
        Leg(id=buy_leg_id, trade_id=id, type="buy", date=buy_date, price=buy_price, quantity=10),
        Leg(id=sell_leg_id, trade_id=id, type="sell", date=sell_date, price=sell_price, quantity=10),
    ]
    return trade


REFERENCE_TRADES: dict[str, Trade] = {
    t.id: t
    for t in [
        _trade("t1", "AAPL", 100, "2024-01-01", 110, "2024-01-10", "l1", "l2"),
        _trade("t2", "AAPL", 100, "2024-02-01", 90, "2024-02-10", "l3", "l4"),
        _trade("t3", "TSLA", 200, "2024-03-01", 250, "2024-03-15", "l5", "l6"),
    ]
}

REFERENCE_SCORES: list[TimingScore] = [
    TimingScore(leg_id="l1", window_days=10, local_low=90, local_high=120, percentile=40.0, verdict="neutral"),
    TimingScore(leg_id="l2", window_days=10, local_low=90, local_high=120, percentile=20.0, verdict="weak"),
    TimingScore(leg_id="l3", window_days=10, local_low=85, local_high=115, percentile=60.0, verdict="neutral"),
    TimingScore(leg_id="l4", window_days=10, local_low=85, local_high=115, percentile=15.0, verdict="weak"),
    TimingScore(leg_id="l5", window_days=10, local_low=190, local_high=260, percentile=10.0, verdict="strong"),
    TimingScore(leg_id="l6", window_days=10, local_low=190, local_high=260, percentile=92.0, verdict="strong"),
]

REFERENCE_CONCEPTS = ["Disposition Effect", "Loss Aversion", "Overtrading"]


# -- 1. Critic precision/recall against labeled synthetic claims ------------


def load_golden_claims() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_CLAIMS_PATH.read_text().splitlines() if line.strip()]


def _claim_to_coach_output(case: dict) -> CoachOutput:
    if case["type"] == "trade_verdict":
        return CoachOutput(
            trade_verdicts=[TradeVerdict(trade_id=case["trade_id"], verdict_text="synthetic claim", cited_score_ids=case["cited_score_ids"])]
        )
    return CoachOutput(
        patterns=[
            PatternClaim(description=case["description"], evidence_trade_ids=case["evidence_trade_ids"], cited_concept=case["cited_concept"])
        ]
    )


def evaluate_critic_precision_recall() -> dict:
    """Positive class = 'this claim is actually ungrounded'. Precision:
    of the claims the Critic rejected, how many were truly ungrounded.
    Recall: of the truly ungrounded claims, how many did the Critic catch."""
    tp = fp = fn = tn = 0
    details = []

    for case in load_golden_claims():
        coach_output = _claim_to_coach_output(case)
        result = verify_claims(coach_output, list(REFERENCE_TRADES.values()), REFERENCE_SCORES, REFERENCE_CONCEPTS)
        actually_rejected = bool(result.rejected)
        expected_rejected = case["expected_rejected"]

        if expected_rejected and actually_rejected:
            tp += 1
        elif not expected_rejected and actually_rejected:
            fp += 1
        elif expected_rejected and not actually_rejected:
            fn += 1
        else:
            tn += 1

        details.append({"name": case["name"], "expected_rejected": expected_rejected, "actually_rejected": actually_rejected})

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "details": details}


# -- 2. LLM-as-judge over narrative groundedness ----------------------------


class NarrativeJudgment(BaseModel):
    claim_id: str
    grounded: bool
    explanation: str


class GroundednessJudgment(BaseModel):
    judgments: list[NarrativeJudgment] = Field(default_factory=list)


JUDGE_SYSTEM_PROMPT = """\
You are grading a trading coach's narrative claims for groundedness. For \
each claim below, you are shown the exact computed timing scores it cites. \
Mark grounded=true only if the claim's prose is a faithful description of \
those numbers (e.g. it doesn't call a neutral-percentile entry "textbook \
timing", and doesn't claim a trend the numbers don't support). This is a \
narrative-accuracy check, not a citation-existence check -- assume all \
citations already point to real data.
"""


def _format_claim_for_judge(claim_id: str, text: str, cited_scores: list[TimingScore]) -> str:
    score_lines = "\n".join(
        f"  - percentile={s.percentile:.1f} verdict={s.verdict} (range [{s.local_low}, {s.local_high}])" for s in cited_scores
    )
    return f"Claim {claim_id}: \"{text}\"\nCited scores:\n{score_lines or '  (none)'}"


def build_judge_prompt(coach_output: CoachOutput, timing_scores: list[TimingScore]) -> str:
    scores_by_leg_id = {s.leg_id: s for s in timing_scores}
    blocks = []
    for verdict in coach_output.trade_verdicts:
        cited = [scores_by_leg_id[sid] for sid in verdict.cited_score_ids if sid in scores_by_leg_id]
        blocks.append(_format_claim_for_judge(f"trade:{verdict.trade_id}", verdict.verdict_text, cited))
    return "\n\n".join(blocks) if blocks else "(no trade verdicts to judge)"


def judge_groundedness(coach_output: CoachOutput, timing_scores: list[TimingScore], llm_client: LLMClient) -> GroundednessJudgment:
    prompt = build_judge_prompt(coach_output, timing_scores)
    return llm_client.generate_structured(prompt, GroundednessJudgment, system=JUDGE_SYSTEM_PROMPT)


if __name__ == "__main__":
    report = evaluate_critic_precision_recall()
    print(f"Critic precision: {report['precision']:.2f}  recall: {report['recall']:.2f}")
    print(f"tp={report['tp']} fp={report['fp']} fn={report['fn']} tn={report['tn']}")
    for detail in report["details"]:
        status = "OK" if detail["expected_rejected"] == detail["actually_rejected"] else "MISMATCH"
        print(f"  [{status}] {detail['name']}: expected_rejected={detail['expected_rejected']} actual={detail['actually_rejected']}")
