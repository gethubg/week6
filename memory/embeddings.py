"""Text embedding abstraction shared by rag/ and memory/pattern_store.py.

HashEmbedder is a dependency-free, deterministic bag-of-words hashing-trick
embedder: no API key, no network, and (unlike a random/hash-of-whole-string
embedding) texts that share vocabulary actually end up with higher cosine
similarity, which is what the concept-KB threshold logic needs to be
meaningfully testable offline. Swap in OpenAIEmbedder for real semantic
embeddings once OPENAI_API_KEY is available.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder:
    """Real path: text-embedding-3-small, 1536 dims (Tech Spec §3.2)."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model

    def embed(self, text: str) -> list[float]:
        import openai  # lazy import: keeps this module importable without the package/key

        client = openai.OpenAI()
        response = client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One request for N texts instead of N requests -- the embeddings
        endpoint accepts a list `input` and returns embeddings in the same
        order, so this is a straight round-trip reduction, not an
        approximation of embed()."""
        import openai  # lazy import: keeps this module importable without the package/key

        if not texts:
            return []
        client = openai.OpenAI()
        response = client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in response.data]


class NebiusEmbedder:
    """Real path via Nebius AI Studio's Token Factory: an OpenAI-compatible
    embeddings endpoint, so this reuses the `openai` SDK with a custom
    base_url."""

    _BASE_URL = "https://api.studio.nebius.com/v1/"

    def __init__(self, model: str = "Qwen/Qwen3-Embedding-8B") -> None:
        self._model = model

    def embed(self, text: str) -> list[float]:
        import openai  # lazy import: keeps this module importable without the package/key

        client = openai.OpenAI(base_url=self._BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
        response = client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import openai  # lazy import: keeps this module importable without the package/key

        if not texts:
            return []
        client = openai.OpenAI(base_url=self._BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
        response = client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in response.data]


class HashEmbedder:
    """Test/offline stand-in: hashing-trick bag-of-words, L2-normalized."""

    def __init__(self, dims: int = 256) -> None:
        self._dims = dims

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dims
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self._dims
            vector[bucket] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Pure local compute, no network round-trip to save -- looping is
        # already as cheap as it gets.
        return [self.embed(t) for t in texts]
