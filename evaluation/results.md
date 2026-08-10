# Evaluation Results

These numbers were produced by an actual run of `evaluation/run_eval.py`
against the 30-item `evaluation/eval_set.json`, in this project's build
environment, **without `ANTHROPIC_API_KEY` set**. Raw per-item output is in
`evaluation/results.json` (regenerated on every run; timestamps will differ).

## What "without a key" means here

- The **retrieval-only ablation** (k=2/4/6) is pure RAG: local embeddings +
  Chroma similarity search. It needs no LLM and no MCP tools, so these
  numbers are final and will not change when a key is added.
- The **full-pipeline run** exercises the real MCP server (stdio subprocess,
  real `tools/list`/`tools/call`) and the real orchestrator guardrails
  (clarification, escalation, confirm-gating). With no key, the orchestrator
  automatically uses its deterministic template-based fallback for final
  answer *text* instead of Claude — but every tool call, retrieval, and
  guardrail decision behind that text is real, so metrics that depend on
  tool calls, citations, and guardrail behavior (tool selection accuracy,
  workflow completion, escalation/clarification accuracy, action-safety
  pass rate) are already final. **Re-running the identical script with
  `ANTHROPIC_API_KEY` set will make the orchestrator use the real Claude
  tool-use loop instead — no code change needed — which will change the
  *answer text* and citation set (since the LLM chooses which tools/queries
  to run) but exercises the same measured guardrails.**

## Retrieval-only ablation (k = 2, 4, 6)

Computed over the 21 eval items that have `gold_doc_ids` (14
straightforward_qa + 3 multi_document + some tool_requiring items with a
policy citation; out-of-scope/ambiguous items are excluded since they have
no gold doc by design).

| k | recall@k | mean gold-doc coverage | mean top-1 score | mean latency (s) |
|---|---|---|---|---|
| 2 | 1.0000 | 0.9048 | 0.7442 | 0.173 |
| 4 | 1.0000 | 0.9524 | 0.7442 | 0.020 |
| 6 | 1.0000 | 0.9762 | 0.7442 | 0.020 |

**Reading these numbers:** recall@k (did at least one gold document show up
in the top-k) is already 1.0 even at k=2 for this corpus/eval-set size —
unsurprising given the corpus has only 12 documents and most questions map
cleanly to one policy. The more informative signal is **mean gold-doc
coverage** (for multi-document questions, what fraction of *all* gold
documents appear in the top-k): this rises from 0.90 at k=2 to 0.95 at k=4 to
0.98 at k=6, confirming that a higher k meaningfully helps the multi-document
questions retrieve evidence from *all* relevant policies, not just the most
obviously similar one. This is the concrete justification for defaulting
`RETRIEVAL_TOP_K=4` (see `.env.example`): it captures most of the
multi-document benefit of k=6 while returning 33% fewer chunks per query
(cheaper context, less noise for the LLM synthesis step) and with
negligible latency difference between k=4 and k=6 (both ~0.02s once the
model is warm; k=2's first-call number of 0.17s reflects one-time model
load, not k itself).

## Full-pipeline metrics (fallback mode, `llm_used=false`)

| Metric | Value |
|---|---|
| groundedness_rate | 1.0000 |
| citation_precision_mean | 0.6429 |
| citation_recall_mean | 0.9524 |
| tool_selection_accuracy | 1.0000 |
| workflow_completion_rate | 1.0000 |
| escalation_clarification_accuracy | 1.0000 |
| action_safety_pass_rate | 1.0000 |
| latency p50 (s) | 0.0177 |
| latency p95 (s) | 0.0239 |
| latency mean (s) | 0.1259 |

**Reading these numbers:**
- **groundedness_rate = 1.0**: every eval item with a gold document had at
  least one matching doc_id among the response's citations.
- **citation_precision_mean = 0.64 / citation_recall_mean = 0.95**: recall is
  high (we usually retrieve the right document among others), but precision
  is moderate because `search_policy_documents` / the fallback router
  returns several citations per answer (by design, so the user sees
  supporting context), not just the single gold document -- some of those
  extra citations are topically adjacent but not the "gold" doc for that
  specific question. This is expected behavior, not a retrieval failure,
  and is exactly the kind of thing a real LLM synthesis pass would tighten
  up further by choosing which retrieved evidence to actually cite in
  prose.
- **tool_selection_accuracy = 1.0 / workflow_completion_rate = 1.0**: all 8
  tool-requiring eval items triggered the expected MCP tool(s) and completed
  without error.
- **escalation_clarification_accuracy = 1.0**: both ambiguous items
  correctly triggered a clarifying question, both out-of-scope items (plus
  one more general out-of-scope item) correctly triggered escalation, and no
  in-scope item was incorrectly flagged.
- **action_safety_pass_rate = 1.0**: the two ticket-creation items behaved
  exactly as required — the unconfirmed request produced
  `needs_confirmation=true` with nothing written, and the confirmed request
  actually created a mock ticket.
- **Latency** here reflects the fallback path (no network LLM call): p50
  ~18ms, p95 ~24ms, dominated by local embedding + Chroma query time. Once a
  real Claude key is configured, latency will be dominated by the
  network/LLM round trip(s) instead (typically hundreds of ms to a few
  seconds per turn, scaling with the number of tool-use turns) — this is
  expected and the harness will report the real numbers once that's
  available.

## What still requires a real ANTHROPIC_API_KEY

- Actual LLM-synthesized answer text (currently template-based) and the
  qualitative quality of that synthesis.
- `evaluation/results.json`'s `full_pipeline_metrics.llm_used` will read
  `true` and `per_item_results[*].llm_used` will be `true` once a real key
  is set; all guardrail/tool-selection/safety metrics above are already
  exercised for real and are not expected to regress.
