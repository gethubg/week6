"""Deterministic eval layer (Tech Spec §11): golden_trades.jsonl has
hand-computed local_low/local_high/percentile/verdict for a handful of
price windows. This checks the Analysis Agent's math (mcp_server.tools +
tools.timing_score) directly, independent of any LLM call.

Run standalone: uv run python -m evals.deterministic_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp_server.tools import compute_local_extremes
from tools.timing_score import timing_percentile, verdict_for

GOLDEN_PATH = Path(__file__).parent / "golden_trades.jsonl"
TOLERANCE = 0.01


def load_golden_cases() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]


def check_case(case: dict) -> list[str]:
    """Returns a list of human-readable mismatches; empty means the case passed."""
    mismatches: list[str] = []

    extremes = compute_local_extremes(case["bars"], case["date"], case["window_days"])
    if abs(extremes["local_low"] - case["expected_local_low"]) > TOLERANCE:
        mismatches.append(f"local_low {extremes['local_low']} != expected {case['expected_local_low']}")
    if abs(extremes["local_high"] - case["expected_local_high"]) > TOLERANCE:
        mismatches.append(f"local_high {extremes['local_high']} != expected {case['expected_local_high']}")

    percentile = timing_percentile(case["execution_price"], extremes["local_low"], extremes["local_high"])
    if abs(percentile - case["expected_percentile"]) > TOLERANCE:
        mismatches.append(f"percentile {percentile} != expected {case['expected_percentile']}")

    verdict = verdict_for(percentile, case["side"])
    if verdict != case["expected_verdict"]:
        mismatches.append(f"verdict {verdict!r} != expected {case['expected_verdict']!r}")

    return mismatches


def run_deterministic_eval() -> dict[str, list[str]]:
    return {case["name"]: check_case(case) for case in load_golden_cases()}


if __name__ == "__main__":
    results = run_deterministic_eval()
    failed = {name: mismatches for name, mismatches in results.items() if mismatches}

    for name, mismatches in results.items():
        print(f"[{'FAIL' if mismatches else 'PASS'}] {name}")
        for mismatch in mismatches:
            print(f"    {mismatch}")

    print(f"\n{len(results) - len(failed)}/{len(results)} golden cases passed")
    raise SystemExit(1 if failed else 0)
