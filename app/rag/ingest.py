"""
Ingestion pipeline: corpus files -> parsed sections -> heading-aware chunks
-> local embeddings -> persistent Chroma collection.

Chroma (persistent, local, on-disk at CHROMA_PERSIST_DIR) was chosen as the
vector store because it needs no separate server process or cloud account,
persists to a simple on-disk directory (fits the single free-tier deployment
constraint), and has a small, stable Python API that supports storing
arbitrary per-chunk metadata alongside vectors -- which we rely on for
citations (doc_id, title, section heading, source format).

Rebuilding is idempotent: `build_index(reset=True)` (the default, and what
scripts/build_index.py uses) deletes and recreates the collection each time,
so re-running the script after editing the corpus always yields a consistent
index with no stale or duplicate chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import settings
from app.rag.chunking import chunk_sections
from app.rag.embeddings import embed_texts
from app.rag.parsers import iter_corpus_files, parse_document


@dataclass
class IngestStats:
    documents: int
    chunks: int
    doc_ids: list[str]


def _get_client() -> chromadb.PersistentClient:
    persist_dir = settings.resolve(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(client: chromadb.PersistentClient | None = None, create: bool = True):
    client = client or _get_client()
    if create:
        return client.get_or_create_collection(name=settings.chroma_collection_name)
    return client.get_collection(name=settings.chroma_collection_name)


def build_index(
    corpus_dir: str | Path | None = None,
    reset: bool = True,
) -> IngestStats:
    """(Re)build the Chroma index from the markdown/html/txt corpus.

    Returns IngestStats with counts, useful for logging and for tests that
    assert the index actually contains chunks after a build.
    """
    corpus_path = settings.resolve(str(corpus_dir)) if corpus_dir else settings.resolve(settings.corpus_dir)

    client = _get_client()
    if reset:
        try:
            client.delete_collection(settings.chroma_collection_name)
        except Exception:
            pass
    collection = get_collection(client, create=True)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    doc_ids: list[str] = []

    for path in iter_corpus_files(corpus_path):
        parsed = parse_document(path)
        doc_ids.append(parsed.doc_id)
        chunks = chunk_sections(parsed.sections)
        for chunk in chunks:
            chunk_id = f"{parsed.doc_id}::{chunk.order}::{chunk.chunk_index}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "doc_id": parsed.doc_id,
                    "title": parsed.title,
                    "section": chunk.heading,
                    "source_format": parsed.source_format,
                    "source_path": str(path.relative_to(corpus_path)),
                    "chunk_order": chunk.order,
                    "chunk_index": chunk.chunk_index,
                }
            )

    if not ids:
        return IngestStats(documents=0, chunks=0, doc_ids=[])

    embeddings = embed_texts(documents)

    # Chroma add() has a practical batch-size ceiling; batch defensively.
    batch_size = 256
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    return IngestStats(documents=len(doc_ids), chunks=len(ids), doc_ids=doc_ids)
