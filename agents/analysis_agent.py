"""Analysis Agent (Product Spec §5.2 / Tech Spec §6.2).

For each buy/sell leg of a trade, calls the MCP price tool for local
extremes and scores timing via the pure-Python ``timing_score`` tool --
deterministic math lives in tools, not LLM reasoning. Dividend legs are
skipped (no buy/sell side to score). On a price-data error, the leg is
skipped and the error recorded rather than fabricating a score.
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from langsmith import traceable

from db.db import insert_timing_score
from db.models import Trade, TimingScore
from mcp_server.client import get_local_extremes
from tools.timing_score import timing_percentile, verdict_for

ExtremesFn = Callable[[str, str, int], dict]

DEFAULT_WINDOW_DAYS = 10


@traceable(name="analysis_agent", run_type="chain")
def score_trades(
    trades: list[Trade],
    conn: sqlite3.Connection,
    extremes_fn: ExtremesFn = get_local_extremes,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> tuple[list[TimingScore], list[str]]:
    scores: list[TimingScore] = []
    errors: list[str] = []

    for trade in trades:
        for leg in trade.legs:
            if leg.type not in ("buy", "sell"):
                continue

            result = extremes_fn(trade.ticker, leg.date, window_days)
            if "error" in result:
                errors.append(f"{trade.ticker} {leg.date}: {result['error']}")
                continue

            percentile = timing_percentile(leg.price, result["local_low"], result["local_high"])
            verdict = verdict_for(percentile, leg.type)
            score = TimingScore(
                leg_id=leg.id,
                window_days=window_days,
                local_low=result["local_low"],
                local_high=result["local_high"],
                percentile=percentile,
                verdict=verdict,
            )
            scores.append(score)
            insert_timing_score(conn, score)

    conn.commit()
    return scores, errors
