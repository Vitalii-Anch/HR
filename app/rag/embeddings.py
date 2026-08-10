"""
Local embedding model loader.

Design decision (documented here and in design-and-evaluation.md):
We use sentence-transformers' `all-MiniLM-L6-v2` -- a small (~90MB), fast,
free, locally-run embedding model that needs no API key and produces
384-dimensional embeddings that work well for short-to-medium policy-document
chunks. This keeps the RAG pipeline fully usable even when no LLM API key is
configured (a hard requirement of this project).

Loading strategy (in order of preference), all offline-friendly:
  1. If EMBEDDING_MODEL_PATH already contains a valid sentence-transformers
     model directory (config.json + weights), load directly from disk.
  2. Otherwise, if the `all-minilm-l6-v2-model` pip package is installed
     (a normal PyPI wheel that bundles the official
     sentence-transformers/all-MiniLM-L6-v2 weights), extract it once to
     EMBEDDING_MODEL_PATH and load from there. This lets `pip install -r
     requirements.txt` alone produce a fully working, offline embedding
     pipeline with no dependency on huggingface.co being reachable at
     runtime -- useful both for restricted sandboxes and for fast, reliable
     cold starts on a free-tier deployment host.
  3. As a last resort, fall back to loading EMBEDDING_MODEL_NAME directly
     from the Hugging Face Hub (requires network access).

The loaded model is cached at module level (`_MODEL`) so repeated calls
within a process (ingestion, then many retrieval calls) do not reload it.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

from app.config import settings

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _load_model():
    from sentence_transformers import SentenceTransformer

    model_path = settings.resolve(settings.embedding_model_path)
    marker = model_path / "config.json"

    if marker.exists():
        return SentenceTransformer(str(model_path))

    # Try extracting from the bundled pip package (offline, no network needed).
    try:
        from all_minilm_l6_v2 import extract_model

        model_path.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_model(str(model_path.parent))
        return SentenceTransformer(str(extracted))
    except Exception:
        pass

    # Last resort: download from the Hugging Face Hub by model name.
    return SentenceTransformer(settings.embedding_model_name)


def get_embedding_model():
    """Return the process-wide cached SentenceTransformer instance (thread-safe, lazy)."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = _load_model()
    return _MODEL


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Embed a batch of texts and return L2-normalized embedding vectors as plain lists."""
    model = get_embedding_model()
    texts = list(texts)
    if not texts:
        return []
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
