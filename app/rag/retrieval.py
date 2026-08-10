"""
Query-time retrieval: embed the query, run top-k similarity search against
the Chroma collection, and (optionally) apply a simple keyword-overlap boost
before returning the final top-k results with citation metadata.

Retrieval k defaults to RETRIEVAL_TOP_K (default 4), matching the project's
env-based configuration so the same value used at build/eval time is used by
the agent at query time. See evaluation/run_eval.py for an ablation across
k=2/4/6 on retrieval-only metrics.

Keyword-boost reranking rationale: dense embedding similarity alone can
occasionally rank a topically-related but less specific chunk above one that
contains an exact term the user asked about (e.g. "blackout period"). A small
lexical-overlap boost, computed cheaply via bag-of-words intersection, nudges
chunks that literally contain the user's query terms slightly higher without
overriding semantic similarity -- a common, low-cost improvement over
embedding-only retrieval that doesn't require a second model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.rag.embeddings import embed_query
from app.rag.ingest import get_collection

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    section: str
    text: str
    source_format: str
    source_path: str
    similarity: float
    keyword_boost: float
    score: float
    metadata: dict = field(default_factory=dict)

    def citation(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "section": self.section}


def retrieve(
    query: str,
    k: int | None = None,
    keyword_boost_weight: float = 0.15,
    fetch_multiplier: int = 3,
) -> list[RetrievedChunk]:
    """Return the top-k chunks for `query`, ranked by similarity + keyword boost."""
    k = k or settings.retrieval_top_k
    collection = get_collection(create=True)

    count = collection.count()
    if count == 0:
        return []

    query_embedding = embed_query(query)
    fetch_n = min(max(k * fetch_multiplier, k), count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    query_tokens = _tokenize(query)

    candidates: list[RetrievedChunk] = []
    for doc_text, meta, dist in zip(docs, metas, dists):
        # Chroma's default distance is squared L2 on normalized embeddings;
        # convert to a similarity-like score in roughly [0, 1] (higher is better).
        similarity = max(0.0, 1.0 - (dist / 2.0))
        doc_tokens = _tokenize(doc_text)
        overlap = len(query_tokens & doc_tokens)
        keyword_boost = keyword_boost_weight * (overlap / max(len(query_tokens), 1))
        score = similarity + keyword_boost
        candidates.append(
            RetrievedChunk(
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                section=meta.get("section", ""),
                text=doc_text,
                source_format=meta.get("source_format", ""),
                source_path=meta.get("source_path", ""),
                similarity=similarity,
                keyword_boost=keyword_boost,
                score=score,
                metadata=meta,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k]


def format_citations(chunks: list[RetrievedChunk]) -> list[dict]:
    """De-duplicate to one citation per (doc_id, section), preserving order."""
    seen = set()
    citations = []
    for c in chunks:
        key = (c.doc_id, c.section)
        if key in seen:
            continue
        seen.add(key)
        citations.append(c.citation())
    return citations
