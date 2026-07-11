"""Create the Pinecone serverless indexes referenced by PINECONE_INDEX_NEWS /
_CONCEPTS / _PATTERNS if they don't already exist (Tech Spec §3.2).

Idempotent: safe to run repeatedly. Dimension must match whatever embedder
app.pipeline.make_embedder() picks at runtime (Nebius/OpenAI key presence
decides which one) -- these are unrelated at import time, so pass the
embedder explicitly rather than re-deriving the same precedence here.

Usage:
    python -m memory.create_index
"""

from __future__ import annotations

import os
import time

INDEX_ENV_VARS = ["PINECONE_INDEX_NEWS", "PINECONE_INDEX_CONCEPTS", "PINECONE_INDEX_PATTERNS"]


def create_indexes(dimension: int) -> None:
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is not set -- add it to your .env first.")

    names = [os.environ[var] for var in INDEX_ENV_VARS if os.environ.get(var)]
    if not names:
        raise SystemExit(f"None of {INDEX_ENV_VARS} are set -- add them to your .env first.")

    pc = Pinecone(api_key=api_key)
    existing = {ix["name"] for ix in pc.list_indexes()}
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    for name in names:
        if name in existing:
            print(f"Index '{name}' already exists -- nothing to do.")
            continue

        print(f"Creating serverless index '{name}' (dim={dimension}, metric=cosine, {cloud}/{region})...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
        print(f"Index '{name}' is ready.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    from app.pipeline import make_embedder

    probe_dimension = len(make_embedder().embed("dimension probe"))
    create_indexes(probe_dimension)
