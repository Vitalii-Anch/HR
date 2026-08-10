#!/usr/bin/env python3
"""
Evaluation harness for the HR Agentic RAG system.

Runs evaluation/eval_set.json through the pipeline and reports:
  - groundedness (heuristic: gold-doc-id overlap with retrieved/cited docs)
  - citation accuracy (precision/recall of citations vs. gold_doc_ids)
  - tool selection accuracy (did the right MCP tools get called)
  - workflow completion rate (did tool-requiring items finish without error)
  - escalation/clarification accuracy (ambiguous -> clarify, out-of-scope -> escalate)
  - action-safety pass rate (mock ticket creation only ever executes with confirm=true)
  - latency p50 / p95 (wall-clock per item)

Two modes, chosen automatically based on whether ANTHROPIC_API_KEY is set:
  - No key (this project's build/CI environment): the orchestrator runs its
    deterministic fallback path. All of the above metrics are still computed
    for real against that fallback path -- this is not a mock run of the
    eval harness itself, it is a real run of a mode that doesn't need an LLM.
  - Key set: the orchestrator runs the real Claude tool-use loop instead, and
    the exact same eval script / metrics apply to real LLM-synthesized
    answers. No code change is needed to go from one mode to the other.

Also runs a retrieval-only ablation across RETRIEVAL_TOP_K = 2, 4, 6 (pure
RAG, no LLM, no MCP) and reports recall@k / coverage@k / mean top-1 score for
each -- this part is always fully real, with no key required.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --out evaluation/results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.mcp_client import MCPToolClient
from app.agent.orchestrator import Orchestrator
from app.rag.retrieval import retrieve

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Retrieval-only ablation (k = 2, 4, 6) -- always runs for real, no LLM needed.
# ---------------------------------------------------------------------------

def run_retrieval_ablation(eval_set: list[dict], k_values: list[int] = (2, 4, 6)) -> dict:
    items = [i for i in eval_set if i.get("gold_doc_ids")]
    ablation = {}
    for k in k_values:
        recall_hits = 0
        coverage_fractions = []
        top1_scores = []
        latencies = []
        for item in items:
            start = time.perf_counter()
            results = retrieve(item["question"], k=k)
            latencies.append(time.perf_counter() - start)

            gold = set(item["gold_doc_ids"])
            retrieved_docs = {r.doc_id for r in results}
            hit = len(gold & retrieved_docs) > 0
            recall_hits += int(hit)
            coverage_fractions.append(len(gold & retrieved_docs) / len(gold) if gold else 0.0)
            top1_scores.append(results[0].score if results else 0.0)

        ablation[f"k={k}"] = {
            "n_items": len(items),
            "recall_at_k": round(recall_hits / len(items), 4) if items else None,
            "mean_gold_doc_coverage": round(statistics.mean(coverage_fractions), 4) if coverage_fractions else None,
            "mean_top1_score": round(statistics.mean(top1_scores), 4) if top1_scores else None,
            "mean_latency_sec": round(statistics.mean(latencies), 4) if latencies else None,
        }
    return ablation


# ---------------------------------------------------------------------------
# Full pipeline evaluation (RAG + MCP tools + orchestrator, LLM if configured)
# ---------------------------------------------------------------------------

async def run_full_eval(eval_set: list[dict]) -> dict:
    async with MCPToolClient() as client:
        orch = Orchestrator(client)
        llm_used = orch.llm_client.is_configured()

        per_item = []
        latencies = []

        for item in eval_set:
            start = time.perf_counter()
            try:
                result = await orch.handle_message(
                    item["question"],
                    employee_id=item.get("employee_id"),
                    confirm=bool(item.get("confirm", False)),
                )
                error = None
            except Exception as exc:  # pragma: no cover - defensive
                result = None
                error = str(exc)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            record = {"id": item["id"], "category": item["category"], "latency_sec": round(elapsed, 4)}
            if error:
                record["error"] = error
                per_item.append(record)
                continue

            cited_doc_ids = {c["doc_id"] for c in result.citations}
            tool_names_called = [t.tool_name for t in result.tool_trace]

            record.update(
                {
                    "answer_preview": result.answer[:160],
                    "citations": sorted(cited_doc_ids),
                    "tool_trace_names": tool_names_called,
                    "needs_confirmation": result.needs_confirmation,
                    "clarification_needed": result.clarification_needed,
                    "escalated": result.escalated,
                    "llm_used": result.llm_used,
                }
            )
            per_item.append(record)

        return {"llm_used": llm_used, "items": per_item, "latencies": latencies}


def compute_metrics(eval_set: list[dict], run_output: dict) -> dict:
    items_by_id = {i["id"]: i for i in eval_set}
    records = {r["id"]: r for r in run_output["items"]}

    # Groundedness / citation accuracy (only meaningful for items with gold_doc_ids).
    citation_items = [i for i in eval_set if i.get("gold_doc_ids")]
    groundedness_hits = 0
    precisions, recalls = [], []
    for item in citation_items:
        rec = records.get(item["id"], {})
        gold = set(item["gold_doc_ids"])
        got = set(rec.get("citations", []))
        hit = len(gold & got) > 0
        groundedness_hits += int(hit)
        precisions.append(len(gold & got) / len(got) if got else 0.0)
        recalls.append(len(gold & got) / len(gold) if gold else 0.0)

    # Tool selection accuracy (only for tool_requiring items with expected_tools).
    tool_items = [i for i in eval_set if i.get("expected_tools") is not None]
    tool_hits = 0
    for item in tool_items:
        rec = records.get(item["id"], {})
        expected = set(item["expected_tools"])
        got = set(rec.get("tool_trace_names", []))
        if expected.issubset(got):
            tool_hits += 1

    # Workflow completion rate (tool_requiring items that ran without error and produced an answer).
    workflow_items = [i for i in eval_set if i["category"] == "tool_requiring"]
    completed = sum(1 for i in workflow_items if "error" not in records.get(i["id"], {"error": True}))

    # Escalation / clarification accuracy across the whole eval set.
    correct_flags = 0
    for item in eval_set:
        rec = records.get(item["id"], {})
        expected_clarify = bool(item.get("expects_clarification", False))
        expected_escalate = bool(item.get("expects_escalation", False))
        actual_clarify = bool(rec.get("clarification_needed", False))
        actual_escalate = bool(rec.get("escalated", False))
        if expected_clarify == actual_clarify and expected_escalate == actual_escalate:
            correct_flags += 1

    # Action-safety pass rate: any item touching create_mock_hr_ticket must have
    # needs_confirmation/created behavior consistent with its `requires_confirmation` expectation.
    safety_items = [i for i in eval_set if "requires_confirmation" in i]
    safety_pass = 0
    for item in safety_items:
        rec = records.get(item["id"], {})
        expected_needs_confirm = bool(item["requires_confirmation"])
        actual_needs_confirm = bool(rec.get("needs_confirmation", False))
        if expected_needs_confirm == actual_needs_confirm:
            safety_pass += 1

    latencies = run_output["latencies"]
    sorted_lat = sorted(latencies)

    def percentile(data, p):
        if not data:
            return None
        idx = min(int(round(p / 100 * (len(data) - 1))), len(data) - 1)
        return round(data[idx], 4)

    return {
        "llm_used": run_output["llm_used"],
        "n_items": len(eval_set),
        "groundedness_rate": round(groundedness_hits / len(citation_items), 4) if citation_items else None,
        "citation_precision_mean": round(statistics.mean(precisions), 4) if precisions else None,
        "citation_recall_mean": round(statistics.mean(recalls), 4) if recalls else None,
        "tool_selection_accuracy": round(tool_hits / len(tool_items), 4) if tool_items else None,
        "workflow_completion_rate": round(completed / len(workflow_items), 4) if workflow_items else None,
        "escalation_clarification_accuracy": round(correct_flags / len(eval_set), 4),
        "action_safety_pass_rate": round(safety_pass / len(safety_items), 4) if safety_items else None,
        "latency_p50_sec": percentile(sorted_lat, 50),
        "latency_p95_sec": percentile(sorted_lat, 95),
        "latency_mean_sec": round(statistics.mean(latencies), 4) if latencies else None,
    }


async def main_async(out_path: Path) -> None:
    eval_set = load_eval_set()

    print(f"Loaded {len(eval_set)} eval items from {EVAL_SET_PATH}")

    print("\n=== Retrieval-only ablation (k=2/4/6), no LLM/MCP needed ===")
    ablation = run_retrieval_ablation(eval_set)
    for k, m in ablation.items():
        print(f"  {k}: recall@k={m['recall_at_k']}, coverage={m['mean_gold_doc_coverage']}, "
              f"mean_top1_score={m['mean_top1_score']}, latency={m['mean_latency_sec']}s")

    print("\n=== Full pipeline run (RAG + MCP tools + orchestrator) ===")
    run_output = await run_full_eval(eval_set)
    print(f"LLM used for synthesis: {run_output['llm_used']}")
    metrics = compute_metrics(eval_set, run_output)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrieval_ablation": ablation,
        "full_pipeline_metrics": metrics,
        "per_item_results": run_output["items"],
        "notes": [
            "Retrieval ablation (k=2/4/6) is always computed with real, live retrieval -- no LLM required.",
            "Full-pipeline metrics reflect whichever mode the orchestrator ran in: the deterministic "
            "fallback (llm_used=false, used whenever ANTHROPIC_API_KEY is unset) or the real Claude "
            "tool-use loop (llm_used=true, used automatically once ANTHROPIC_API_KEY is set). No code "
            "change is needed to re-run this script with a real key for final LLM-backed numbers.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()
    asyncio.run(main_async(args.out))


if __name__ == "__main__":
    main()
