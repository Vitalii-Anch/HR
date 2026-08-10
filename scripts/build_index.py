#!/usr/bin/env python3
"""
(Re)build the Chroma vector index from the corpus/ directory.

Usage:
    python scripts/build_index.py

Idempotent: safe to re-run any time the corpus changes; it deletes and
recreates the Chroma collection each time (see app/rag/ingest.build_index).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.rag.ingest import build_index


def main() -> int:
    print(f"Corpus dir:   {settings.resolve(settings.corpus_dir)}")
    print(f"Chroma dir:   {settings.resolve(settings.chroma_persist_dir)}")
    print(f"Collection:   {settings.chroma_collection_name}")
    print(f"Embedding:    {settings.embedding_model_name} (local path: {settings.embedding_model_path})")

    start = time.time()
    stats = build_index()
    elapsed = time.time() - start

    print(f"\nIndexed {stats.documents} documents into {stats.chunks} chunks in {elapsed:.1f}s")
    print("Doc IDs:", ", ".join(sorted(stats.doc_ids)))

    if stats.chunks == 0:
        print("WARNING: no chunks were indexed. Check CORPUS_DIR.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
