"""Vector store abstraction (Tech Spec §3.2). Three logical indexes exist in
production (news, concepts, patterns), each its own Pinecone index; each
gets its own VectorStore instance here too, so swapping one from
InMemoryVectorStore to PineconeVectorStore doesn't touch the others.

The `filter` argument mirrors a (small) subset of Pinecone's metadata
filter dict syntax -- equality (`{"field": value}`) and range
(`{"field": {"$gte": ..., "$lte": ...}}`) -- so both implementations
accept the exact same shape.
"""

from __future__ import annotations

import math
from typing import Protocol, TypedDict


class VectorRecord(TypedDict):
    id: str
    score: float
    metadata: dict


class UpsertRecord(TypedDict):
    id: str
    embedding: list[float]
    metadata: dict


class VectorStore(Protocol):
    def upsert(self, namespace: str, id: str, embedding: list[float], metadata: dict) -> None:
        ...

    def upsert_batch(self, namespace: str, records: list[UpsertRecord]) -> None:
        ...

    def query(
        self, namespace: str, query_embedding: list[float], top_k: int, filter: dict | None = None
    ) -> list[VectorRecord]:
        ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches_filter(metadata: dict, filter: dict) -> bool:
    for field, condition in filter.items():
        value = metadata.get(field)
        if isinstance(condition, dict):
            if "$eq" in condition and value != condition["$eq"]:
                return False
            if "$gte" in condition and (value is None or value < condition["$gte"]):
                return False
            if "$lte" in condition and (value is None or value > condition["$lte"]):
                return False
        else:
            if value != condition:
                return False
    return True


class InMemoryVectorStore:
    """Test/offline stand-in for a single Pinecone index."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def upsert(self, namespace: str, id: str, embedding: list[float], metadata: dict) -> None:
        self._data.setdefault(namespace, {})[id] = (embedding, metadata)

    def upsert_batch(self, namespace: str, records: list[UpsertRecord]) -> None:
        # No network cost to save here; batching only matters for the real
        # Pinecone path below.
        for record in records:
            self.upsert(namespace, record["id"], record["embedding"], record["metadata"])

    def query(
        self, namespace: str, query_embedding: list[float], top_k: int, filter: dict | None = None
    ) -> list[VectorRecord]:
        candidates = self._data.get(namespace, {})
        scored: list[VectorRecord] = []
        for id, (embedding, metadata) in candidates.items():
            if filter and not _matches_filter(metadata, filter):
                continue
            scored.append(VectorRecord(id=id, score=_cosine_similarity(query_embedding, embedding), metadata=metadata))
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


class PineconeVectorStore:
    """Real path: one Pinecone index (Tech Spec §3.2). `index_name` should be
    one of PINECONE_INDEX_NEWS / _CONCEPTS / _PATTERNS from the environment."""

    def __init__(self, index_name: str) -> None:
        self._index_name = index_name
        self._index = None

    def _get_index(self):
        if self._index is None:
            import os

            from pinecone import Pinecone  # lazy import: keeps this module importable without the package/key

            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            self._index = pc.Index(self._index_name)
        return self._index

    def upsert(self, namespace: str, id: str, embedding: list[float], metadata: dict) -> None:
        self._get_index().upsert(vectors=[{"id": id, "values": embedding, "metadata": metadata}], namespace=namespace)

    def upsert_batch(self, namespace: str, records: list[UpsertRecord]) -> None:
        """One HTTP call for N vectors instead of N calls -- Pinecone's
        upsert already accepts a vectors list, so the single-record upsert()
        above was leaving this on the table."""
        if not records:
            return
        vectors = [{"id": r["id"], "values": r["embedding"], "metadata": r["metadata"]} for r in records]
        self._get_index().upsert(vectors=vectors, namespace=namespace)

    def query(
        self, namespace: str, query_embedding: list[float], top_k: int, filter: dict | None = None
    ) -> list[VectorRecord]:
        response = self._get_index().query(
            namespace=namespace, vector=query_embedding, top_k=top_k, filter=filter, include_metadata=True
        )
        return [
            VectorRecord(id=match["id"], score=match["score"], metadata=match.get("metadata", {}))
            for match in response["matches"]
        ]
