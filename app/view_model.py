"""Pure functions turning a TradeCoachState into rows/cards for the
Streamlit views (Tech Spec §12). Kept separate from streamlit_app.py so
this is testable without importing Streamlit at all.
"""

from __future__ import annotations

from graph.state import TradeCoachState


def build_scorecard_rows(state: TradeCoachState) -> list[dict]:
    scores_by_leg_id = {s.leg_id: s for s in state.get("timing_scores", [])}
    coach = state.get("coach_narrative")
    verdicts_by_trade_id = {v.trade_id: v for v in coach.trade_verdicts} if coach else {}
    news_by_leg = state.get("news_by_leg", {})

    rows: list[dict] = []
    for trade in state.get("trades", []):
        for leg in trade.legs:
            score = scores_by_leg_id.get(leg.id)
            if score is None:
                continue
            verdict = verdicts_by_trade_id.get(trade.id)
            linked_news = [n.summary or "" for n in news_by_leg.get(leg.id, [])]
            rows.append(
                {
                    "trade_id": trade.id,
                    "ticker": trade.ticker,
                    "status": trade.status,
                    "leg_type": leg.type,
                    "date": leg.date,
                    "price": leg.price,
                    "percentile": score.percentile,
                    "timing_verdict": score.verdict,
                    "coach_verdict": verdict.verdict_text if verdict else None,
                    "news": "; ".join(n for n in linked_news if n) or None,
                }
            )
    return rows


def build_pattern_cards(state: TradeCoachState) -> list[dict]:
    return [
        {
            "description": p.description,
            "confidence": p.confidence,
            "evidence_trade_ids": p.evidence_trade_ids,
            "last_updated": p.last_updated,
        }
        for p in state.get("patterns_retrieved", [])
    ]
