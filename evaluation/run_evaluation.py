"""Run the grounded RAG evaluation through app.py's production pipeline.

Examples
--------
Run against documents already indexed in one chat:
    .\\venv\\Scripts\\python evaluation\\run_evaluation.py --chat-id <CHAT_ID>

Create an isolated evaluation chat, ingest both source PDFs, then evaluate it:
    .\\venv\\Scripts\\python evaluation\\run_evaluation.py --bootstrap \
      --pdf "C:\\Users\\samar\\Downloads\\Provisional Placement Policy-2026.pdf" \
      --pdf "C:\\Cadence\\RAG_Project\\rag_data\\chats\\a6077f36-8d15-4fdb-86f8-a9dd2591d47c\\Assignment4_Report.pdf"

The script deliberately calls ``generate_response`` from app.py, so each
question exercises dense retrieval, BM25, RRF, Jina reranking, Llama
generation, citations, and the app's evidence/conflict checks.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "evaluation" / "questions.json"
RESULTS_PATH = ROOT / "evaluation" / "results.json"
METRICS_PATH = ROOT / "evaluation" / "metrics.json"
ABSTENTION_TEXT = "i couldn't find that information in the uploaded pdfs"
CITATION_PATTERN = re.compile(r"\[S(\d+)\]", re.IGNORECASE)


def load_production_pipeline() -> Any:
    """Load app functions without starting Streamlit's UI code.

    This mirrors the test fixture's AST loading approach. It intentionally
    preserves production constants, imports, and functions, including
    generate_response; only top-level UI rendering is skipped.
    """
    app_path = ROOT / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))
    selected: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
                             ast.AugAssign, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Try)):
            selected.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            selected.append(node)

    module_ast = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    spec = importlib.util.spec_from_loader("app_evaluation_pipeline", loader=None)
    module = importlib.util.module_from_spec(spec)

    # The UI normally establishes these before the code that uses them.
    for key, value in {
        "current_chat_id": None, "current_chat": None, "messages": [],
        "uploaded_files": [], "selected_documents": [], "chat_history": [],
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value

    exec(compile(module_ast, str(app_path), "exec"), module.__dict__)
    module.load_dotenv(ROOT / ".env")
    module.JINA_API_KEY = os.getenv("JINA_API_KEY")
    module.OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", module.OLLAMA_API_URL)
    module.init_database()
    module.migrate_database_schema()
    return module


def is_abstention(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return (
        ABSTENTION_TEXT in text
        or "couldn't find" in text
        or "cannot find" in text
        or "not available in the uploaded" in text
    )


def cited_sources(answer: str, docs: list[Any]) -> tuple[list[str], int, int]:
    """Return cited source files plus valid/total citation counts."""
    references = [int(value) for value in CITATION_PATTERN.findall(answer)]
    sources: list[str] = []
    valid = 0
    for reference in references:
        if 1 <= reference <= len(docs):
            valid += 1
            sources.append(docs[reference - 1].metadata.get("source_file", "Unknown PDF"))
    return sorted(set(sources)), valid, len(references)


def grade(question: dict[str, Any], answer: str, docs: list[Any], conflicts: list[str]) -> dict[str, Any]:
    answer_lower = answer.lower()
    retrieved_sources = sorted({doc.metadata.get("source_file", "Unknown PDF") for doc in docs})
    cited, valid_citations, citation_count = cited_sources(answer, docs)
    expected_sources = set(question["expected_sources"])
    expected_sources_retrieved = expected_sources.issubset(retrieved_sources)
    expected_sources_cited = expected_sources.issubset(cited)
    abstained = is_abstention(answer)

    if question.get("injection"):
        forbidden = question.get("forbidden_terms", [])
        correct = abstained and not any(term.lower() in answer_lower for term in forbidden)
    elif not question["answerable"]:
        correct = abstained
    else:
        correct = all(term.lower() in answer_lower for term in question["required_terms"])

    citation_validity = None if citation_count == 0 else valid_citations / citation_count
    expected_conflict = question.get("expected_conflict")
    conflict_correct = None
    if expected_conflict is not None:
        # The app returns warnings, while the answer must explicitly identify a
        # true conflict only when one exists. These fixtures contain no true one.
        mentions_conflict = "conflict" in answer_lower and "no conflict" not in answer_lower
        conflict_correct = (bool(conflicts) or mentions_conflict) == expected_conflict

    return {
        "answer_correct": correct,
        "abstained": abstained,
        "retrieved_sources": retrieved_sources,
        "cited_sources": cited,
        "expected_sources_retrieved": expected_sources_retrieved,
        "expected_sources_cited": expected_sources_cited if expected_sources else None,
        "citation_validity": citation_validity,
        "conflict_correct": conflict_correct,
    }


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item["status"] == "completed"]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in completed:
        categories[item["category"]].append(item)

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        answerable = [r for r in items if r["answerable"]]
        abstention_cases = [r for r in items if not r["answerable"]]
        citation_cases = [r for r in answerable if r["citation_validity"] is not None]
        return {
            "count": len(items),
            "answer_correctness": rate(sum(r["answer_correct"] for r in items), len(items)),
            "retrieval_success": rate(sum(r["expected_sources_retrieved"] for r in answerable), len(answerable)),
            "source_citation_recall": rate(sum(r["expected_sources_cited"] for r in answerable), len(answerable)),
            "citation_validity": round(mean(r["citation_validity"] for r in citation_cases), 4) if citation_cases else None,
            "abstention_accuracy": rate(sum(r["abstained"] for r in abstention_cases), len(abstention_cases)),
            "mean_latency_s": round(mean(r["latency_s"] for r in items), 4) if items else None,
        }

    answerable = [r for r in completed if r["answerable"]]
    abstention_cases = [r for r in completed if not r["answerable"]]
    injection_cases = [r for r in completed if r["category"] == "prompt_injection"]
    conflict_cases = [r for r in completed if r["conflict_correct"] is not None]
    citation_cases = [r for r in answerable if r["citation_validity"] is not None]
    token_values = [r["tokens"]["total_tokens"] for r in completed if r["tokens"].get("total_tokens") is not None]
    return {
        "status": "completed" if len(completed) == len(results) else "completed_with_errors",
        "question_count": len(results),
        "completed_count": len(completed),
        "failed_count": len(results) - len(completed),
        "answer_correctness": rate(sum(r["answer_correct"] for r in completed), len(completed)),
        "retrieval_success": rate(sum(r["expected_sources_retrieved"] for r in answerable), len(answerable)),
        "source_citation_recall": rate(sum(r["expected_sources_cited"] for r in answerable), len(answerable)),
        "citation_validity": round(mean(r["citation_validity"] for r in citation_cases), 4) if citation_cases else None,
        "evidence_coverage": rate(sum(r["expected_sources_cited"] for r in answerable), len(answerable)),
        "abstention_accuracy": rate(sum(r["abstained"] for r in abstention_cases), len(abstention_cases)),
        "conflict_detection_accuracy": rate(sum(r["conflict_correct"] for r in conflict_cases), len(conflict_cases)),
        "prompt_injection_resistance": rate(sum(r["answer_correct"] for r in injection_cases), len(injection_cases)),
        "mean_latency_s": round(mean(r["latency_s"] for r in completed), 4) if completed else None,
        "total_latency_s": round(sum(r["latency_s"] for r in completed), 4),
        "mean_total_tokens": round(mean(token_values), 2) if token_values else None,
        "total_tokens": sum(token_values) if token_values else None,
        "by_category": {category: summary(items) for category, items in sorted(categories.items())},
    }


def bootstrap_chat(app: Any, pdf_paths: list[Path]) -> str:
    if len(pdf_paths) < 2:
        raise ValueError("--bootstrap requires both controlled PDFs via --pdf.")
    for path in pdf_paths:
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
    chat_id = app.create_chat()
    for path in pdf_paths:
        print(f"Indexing {path.name} into evaluation chat {chat_id}...", flush=True)
        app.process_pdf(chat_id, path.read_bytes(), path.name)
    return chat_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 24 grounded questions through the production RAG pipeline.")
    parser.add_argument("--chat-id", help="Existing chat ID containing both controlled PDFs.")
    parser.add_argument("--bootstrap", action="store_true", help="Create a dedicated chat and index --pdf files before evaluating.")
    parser.add_argument("--pdf", action="append", type=Path, default=[], help="Source PDF path; repeat once per controlled PDF with --bootstrap.")
    parser.add_argument("--limit", type=int, help="Run only the first N questions (useful for a smoke test).")
    parser.add_argument(
        "--start-at", type=int, default=1,
        help="Start at this one-based test number and retain prior saved results.",
    )
    args = parser.parse_args()
    if args.bootstrap == bool(args.chat_id):
        parser.error("Provide exactly one of --chat-id or --bootstrap.")

    os.chdir(ROOT)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not 1 <= args.start_at <= len(questions):
        parser.error(f"--start-at must be between 1 and {len(questions)}.")
    if args.start_at > 1:
        questions = questions[args.start_at - 1:]
    if args.limit is not None:
        questions = questions[:args.limit]
    app = load_production_pipeline()
    chat_id = bootstrap_chat(app, args.pdf) if args.bootstrap else args.chat_id
    assert chat_id is not None

    results: list[dict[str, Any]] = []
    if args.start_at > 1 and RESULTS_PATH.exists():
        saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        results = [item for item in saved if int(item["id"][1:]) < args.start_at]
        if len(results) != args.start_at - 1:
            parser.error(
                f"results.json must contain Tests 1–{args.start_at - 1} before resuming. "
                "Use --start-at 1 to start a fresh run."
            )
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['id']} {question['category']}", flush=True)
        try:
            response = app.generate_response(None, question["question"], [], chat_id)
            answer, _, _, latency, docs, evidence_status, conflicts, stage_timings, tokens, request_id = response
            result = {
                "id": question["id"], "category": question["category"], "question": question["question"],
                "expected_answer": question["expected_answer"], "expected_sources": question["expected_sources"],
                "answerable": question["answerable"], "status": "completed", "answer": answer,
                "request_id": request_id, "latency_s": round(latency, 4), "stage_timings_s": stage_timings,
                "tokens": tokens, "evidence_status": evidence_status, "conflicts": conflicts,
                "final_context_sources": [
                    {"source_file": doc.metadata.get("source_file"), "page": (doc.metadata.get("page") or 0) + 1,
                     "chunk_id": doc.metadata.get("chunk_id"), "jina_score": doc.metadata.get("jina_score")}
                    for doc in docs
                ],
            }
            result.update(grade(question, answer, docs, conflicts))
        except Exception as error:  # Preserve per-question failure evidence instead of losing the run.
            result = {
                "id": question["id"], "category": question["category"], "question": question["question"],
                "expected_answer": question["expected_answer"], "expected_sources": question["expected_sources"],
                "answerable": question["answerable"], "status": "failed", "error": repr(error),
            }
        results.append(result)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = aggregate(results)
    metrics["chat_id"] = chat_id
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if metrics["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
