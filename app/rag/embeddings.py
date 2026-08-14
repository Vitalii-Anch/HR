"""
Local embedding: lightweight TF-IDF sparse-vector embedding, implemented
with pure Python/stdlib (no PyTorch, no ONNX runtime, no model weights to
download or extract).

Design decision -- and why this replaced a prior sentence-transformers/
PyTorch implementation:
The original design used sentence-transformers' `all-MiniLM-L6-v2` via
PyTorch for dense neural embeddings. That worked correctly locally and in
CI, but PyTorch's own resident memory footprint (importing torch +
sentence-transformers reliably adds 200-400MB, independent of the ~90MB
model weights themselves) combined with the rest of the app's baseline
memory reliably exceeded Render's free-tier 512MB container cap the first
time a RAG-backed tool call loaded it inside the MCP server subprocess,
causing a reproducible OOM crash. See `deployed.md` for the production
incident this fixes.

Given a small, topically well-separated 12-document HR policy corpus (~56KB
of text total) and a hard free-tier memory budget, TF-IDF cosine similarity
is a well-understood classical-IR alternative that needs no large model
weights and adds negligible memory (a vocabulary + IDF table serialized as
one small JSON file). The tradeoff is retrieval quality: TF-IDF matches on
literal/lexical term overlap rather than learned semantic similarity, so a
paraphrased query that shares little vocabulary with the relevant chunk
retrieves worse than with dense neural embeddings. See
`evaluation/results.md` for retrieval metrics measured under this design
and `design-and-evaluation.md` for the fuller writeup of this tradeoff.

Same public contract as the previous implementation, so `app/rag/ingest.py`
and `app/rag/retrieval.py` need no changes:
    embed_texts(list[str]) -> list[list[float]]
    embed_query(str) -> list[float]
both returning L2-normalized vectors of a fixed dimension (the fitted
vocabulary size), which Chroma stores and searches exactly like any other
embedding.

Fit/persist model: `embed_texts()` is called exactly once per index build
(by `build_index()`, with the full corpus's chunk texts), which fits the
vocabulary + IDF table and persists them to `_VOCAB_PATH`. `embed_query()`
(called at request time, often from a different OS process -- the MCP
server subprocess -- than the one that built the index) loads that
persisted table lazily and reuses it, so query vectors land in the same
vector space as the indexed document vectors.
"""
from __future__ import annotations

import json
import math
import re
import threading
from typing import Iterable

from app.config import settings

_WORD_RE = re.compile(r"[a-z0-9]+")

# A small, standard English stopword list. Without this, generic connective
# words ("the", "is", "of", "policy", "employee"...) that appear in nearly
# every chunk (including boilerplate "Purpose"/"Overview" sections) dominate
# the dot product between short queries and long chunks, regularly outranking
# the chunk that actually contains the query's distinguishing term. This is
# the standard first-line fix for that failure mode in classical TF-IDF
# retrieval, and was verified in practice to meaningfully improve top-k
# ranking on this corpus (see evaluation/results.md).
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have if in into is it its of on
    or such that the their there these this to was were will with you your
    they them then than but not no can may might must shall should would
    could do does did doing about above after again against all am any
    because been before being below between both down during each few
    further he her here hers herself him himself his how i me more most
    my myself once only other our ours ourselves out over own s same she
    so some t too under until up very we what when where which while who
    whom why itself t also may per within upon
    """.split()
)

_VOCAB_PATH = settings.resolve(settings.chroma_persist_dir).parent / "tfidf_vocab.json"

_VOCAB: dict[str, int] | None = None  # term -> index
_IDF: list[float] | None = None  # index -> idf weight
_LOCK = threading.Lock()


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _fit(texts: list[str]) -> tuple[dict[str, int], list[float]]:
    doc_token_sets = [set(_tokenize(t)) for t in texts]
    vocab_terms = sorted({term for tokens in doc_token_sets for term in tokens})
    vocab = {term: i for i, term in enumerate(vocab_terms)}

    n_docs = len(texts)
    df = [0] * len(vocab_terms)
    for tokens in doc_token_sets:
        for term in tokens:
            df[vocab[term]] += 1

    # Smoothed IDF, matching scikit-learn's default formula:
    # idf(t) = ln((n_docs + 1) / (df(t) + 1)) + 1
    idf = [math.log((n_docs + 1) / (d + 1)) + 1.0 for d in df]
    return vocab, idf


def _vectorize(text: str, vocab: dict[str, int], idf: list[float]) -> list[float]:
    if not vocab:
        return []
    tokens = _tokenize(text)
    counts: dict[int, int] = {}
    for term in tokens:
        idx = vocab.get(term)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    vec = [0.0] * len(idf)
    for idx, count in counts.items():
        vec[idx] = count * idf[idx]

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _save_vocab(vocab: dict[str, int], idf: list[float]) -> None:
    _VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _VOCAB_PATH.write_text(json.dumps({"vocab": vocab, "idf": idf}), encoding="utf-8")


def _load_vocab_from_disk() -> tuple[dict[str, int], list[float]] | None:
    if not _VOCAB_PATH.exists():
        return None
    data = json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))
    return data["vocab"], data["idf"]


def _ensure_vocab_loaded() -> tuple[dict[str, int], list[float]]:
    global _VOCAB, _IDF
    if _VOCAB is None or _IDF is None:
        with _LOCK:
            if _VOCAB is None or _IDF is None:
                loaded = _load_vocab_from_disk()
                if loaded is None:
                    raise RuntimeError(
                        f"No TF-IDF vocabulary found at {_VOCAB_PATH}. "
                        "Run `python scripts/build_index.py` first to fit and persist it."
                    )
                _VOCAB, _IDF = loaded
    return _VOCAB, _IDF


def get_embedding_model():
    """Kept only for backward compatibility with any old caller that expected
    a loadable model object; TF-IDF has no such object, so this is a no-op."""
    return None


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Fit a TF-IDF vocabulary over `texts` (expected to be the *entire*
    corpus's chunk texts, passed once by `build_index()`), persist it to
    disk, and return each text's normalized TF-IDF vector."""
    global _VOCAB, _IDF
    texts = list(texts)
    if not texts:
        return []

    with _LOCK:
        vocab, idf = _fit(texts)
        _VOCAB, _IDF = vocab, idf
        _save_vocab(vocab, idf)

    return [_vectorize(t, vocab, idf) for t in texts]


def embed_query(text: str) -> list[float]:
    """Embed a single query string using the previously fitted (and
    persisted) vocabulary. Terms not seen at index-build time are ignored,
    same as standard TF-IDF out-of-vocabulary handling."""
    vocab, idf = _ensure_vocab_loaded()
    return _vectorize(text, vocab, idf)
