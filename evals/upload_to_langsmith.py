"""One-off script to mirror the golden datasets into LangSmith so
`evaluate()` runs show up alongside the traced agent runs (Tech Spec §11).
Manual, pre-demo step -- no CI needed at course scope. No-ops with a clear
message if LANGCHAIN_API_KEY isn't set.

Run: uv run python -m evals.upload_to_langsmith
"""

from __future__ import annotations

import os

from evals.deterministic_eval import load_golden_cases
from evals.groundedness_eval import load_golden_claims

DATASET_NAME_TRADES = "trade-coach-golden-trades"
DATASET_NAME_CLAIMS = "trade-coach-golden-claims"


def upload() -> None:
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("LANGCHAIN_API_KEY not set -- skipping LangSmith upload. Set it in .env to enable this.")
        return

    from langsmith import Client

    client = Client()

    for dataset_name, cases, input_keys in (
        (DATASET_NAME_TRADES, load_golden_cases(), ("bars", "date", "window_days", "execution_price", "side")),
        (DATASET_NAME_CLAIMS, load_golden_claims(), ("type", "trade_id", "cited_score_ids", "description", "evidence_trade_ids", "cited_concept")),
    ):
        dataset = client.read_dataset(dataset_name=dataset_name) if client.has_dataset(dataset_name=dataset_name) else client.create_dataset(dataset_name)
        for case in cases:
            inputs = {k: case[k] for k in input_keys if k in case}
            outputs = {k: v for k, v in case.items() if k not in input_keys and k != "name"}
            client.create_example(dataset_id=dataset.id, inputs=inputs, outputs=outputs, metadata={"name": case["name"]})
        print(f"Uploaded {len(cases)} cases to LangSmith dataset '{dataset_name}'")


if __name__ == "__main__":
    upload()
