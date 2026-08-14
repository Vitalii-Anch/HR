"""RAG retrieval tests. No LLM/API key required."""
from app.rag.retrieval import retrieve, format_citations


def test_retrieve_returns_relevant_pto_chunk():
    results = retrieve("How many PTO days do full-time employees accrue per month?", k=4)
    assert len(results) > 0
    assert results[0].doc_id == "pto-policy"
    assert results[0].score > 0.3


def test_retrieve_respects_k():
    results = retrieve("remote work policy", k=2)
    assert len(results) <= 2


def test_out_of_scope_query_scores_low():
    # A query with zero lexical overlap with the corpus's vocabulary should
    # return no results at all under TF-IDF retrieval (see the zero-vector
    # short-circuit in app/rag/retrieval.py's retrieve()), rather than a
    # low-but-nonzero score -- there is nothing in the corpus for the query
    # to meaningfully match against.
    results = retrieve("What's the best pizza topping in Chicago?", k=4)
    assert len(results) == 0


def test_format_citations_dedupes():
    results = retrieve("PTO carryover", k=4)
    citations = format_citations(results)
    keys = [(c["doc_id"], c["section"]) for c in citations]
    assert len(keys) == len(set(keys))
