"""Behavioral-finance concept KB (Product Spec §6 / Tech Spec §9.2).

Static corpus, indexed once at setup. At Coach time, retrieval requires a
similarity threshold -- below it, the Coach must describe the behavior
plainly rather than name a (possibly wrong) formal concept.
"""

from __future__ import annotations

from pathlib import Path

from memory.embeddings import Embedder
from memory.vector_store import VectorStore

CONCEPT_KB_DIR = Path(__file__).parent / "concept_kb"
CONCEPTS_NAMESPACE = "concepts"
SIMILARITY_THRESHOLD = 0.75


def load_concept_docs() -> dict[str, str]:
    """title -> full doc text, for every *.md file in concept_kb/."""
    docs: dict[str, str] = {}
    for path in sorted(CONCEPT_KB_DIR.glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("#").strip()
        docs[title] = text
    return docs


def index_concepts(store: VectorStore, embedder: Embedder) -> None:
    for title, text in load_concept_docs().items():
        store.upsert(namespace=CONCEPTS_NAMESPACE, id=title, embedding=embedder.embed(text), metadata={"title": title, "text": text})


def retrieve_concept(
    store: VectorStore,
    embedder: Embedder,
    pattern_description: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> str | None:
    """Returns the matching concept title, or None if nothing clears the
    similarity threshold -- callers must not fall back to inventing one."""
    results = store.query(namespace=CONCEPTS_NAMESPACE, query_embedding=embedder.embed(pattern_description), top_k=1)
    if not results or results[0]["score"] < threshold:
        return None
    return results[0]["metadata"]["title"]
