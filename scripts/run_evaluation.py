"""Evaluation runner: runs evaluation questions and computes metrics.

Usage:
    python scripts/run_evaluation.py [--mode offline|online] [--questions path]

Modes:
    offline  - Uses mock generator and deterministic embeddings (no API key needed)
    online   - Uses real Mistral API and sentence-transformers (requires MISTRAL_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_sample_documents(client, data_dir: str) -> dict[str, str]:
    """Ingest all sample documents and return filename->doc_id mapping."""
    doc_ids = {}
    sample_dir = Path(data_dir)
    if not sample_dir.exists():
        print(f"ERROR: Sample document directory not found: {sample_dir}")
        return doc_ids

    for filepath in sorted(sample_dir.glob("*")):
        if filepath.suffix.lower() in (".md", ".markdown", ".txt", ".pdf"):
            with open(filepath, "rb") as f:
                content = f.read()
            files = {"file": (filepath.name, content, "text/plain")}
            r = client.post("/documents/upload", files=files)
            if r.status_code == 201:
                data = r.json()
                doc_ids[filepath.name] = data["document_id"]
                print(f"  [OK] Ingested {filepath.name} -> {data['document_id']} ({data['chunk_count']} chunks)")
            else:
                print(f"  [FAIL] Failed to ingest {filepath.name}: {r.status_code} {r.text}")
    return doc_ids


def run_evaluation(questions: list[dict], client, doc_ids: dict[str, str]) -> list[dict]:
    """Run all evaluation questions and collect results."""
    results = []
    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_facts = q.get("expected_answer_facts", [])
        expected_sources = q.get("expected_source_files", [])
        requires_abstention = q.get("requires_abstention", False)

        print(f"\n--- {qid}: {question}")

        r = client.post("/query", json={"question": question, "top_k": 5})
        if r.status_code != 200:
            print(f"  ERROR: HTTP {r.status_code}")
            results.append({
                "id": qid,
                "error": True,
                "http_status": r.status_code,
            })
            continue

        data = r.json()
        answer = data.get("answer", "")
        abstained = data.get("abstained", False)
        citations = data.get("citations", [])
        retrieved_chunks = data.get("retrieved_chunks", 0)

        # --- Compute metrics for this question ---

        # Abstention accuracy
        abstention_correct = (abstained == requires_abstention)

        # Fact coverage (for answerable questions)
        facts_found = []
        if not requires_abstention:
            for fact in expected_facts:
                if fact.lower() in answer.lower():
                    facts_found.append(fact)
        fact_coverage = len(facts_found) / len(expected_facts) if expected_facts else 1.0

        # Citation validity: check chunk_ids are present and quoted_text non-empty
        valid_citations = 0
        for cite in citations:
            if cite.get("chunk_id") and cite.get("quoted_text"):
                valid_citations += 1
        citation_validity_rate = valid_citations / len(citations) if citations else (1.0 if abstained else 0.0)

        # Source precision: check if citation filenames match expected sources
        cited_files = {c.get("filename", "") for c in citations}
        expected_files_set = set(expected_sources)
        source_hits = len(cited_files & expected_files_set)
        source_precision = source_hits / len(cited_files) if cited_files else (1.0 if not expected_sources else 0.0)

        # False answer on unanswerable
        false_answer = requires_abstention and not abstained

        result = {
            "id": qid,
            "question": question,
            "requires_abstention": requires_abstention,
            "abstained": abstained,
            "abstention_correct": abstention_correct,
            "fact_coverage": fact_coverage,
            "facts_found": facts_found,
            "expected_facts": expected_facts,
            "citation_count": len(citations),
            "valid_citations": valid_citations,
            "citation_validity_rate": citation_validity_rate,
            "source_precision": source_precision,
            "cited_files": list(cited_files),
            "expected_sources": expected_sources,
            "retrieved_chunks": retrieved_chunks,
            "false_answer": false_answer,
            "answer_preview": answer[:150],
        }
        results.append(result)

        status = "[PASS]" if abstention_correct else "[FAIL]"
        print(f"  {status} abstained={abstained} (expected={requires_abstention})")
        if not requires_abstention:
            print(f"    Fact coverage: {fact_coverage:.0%} ({len(facts_found)}/{len(expected_facts)})")
        print(f"    Citations: {len(citations)} (valid: {valid_citations})")
        print(f"    Answer: {answer[:120]}...")

    return results


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from individual results."""
    total = len(results)
    errors = [r for r in results if r.get("error")]
    valid = [r for r in results if not r.get("error")]

    answerable = [r for r in valid if not r["requires_abstention"]]
    unanswerable = [r for r in valid if r["requires_abstention"]]

    # Abstention precision & recall
    true_abstentions = [r for r in valid if r["requires_abstention"] and r["abstained"]]
    [r for r in valid if not r["requires_abstention"] and r["abstained"]]
    predicted_abstentions = [r for r in valid if r["abstained"]]

    abstention_precision = (
        len(true_abstentions) / len(predicted_abstentions)
        if predicted_abstentions else 1.0
    )
    abstention_recall = (
        len(true_abstentions) / len(unanswerable)
        if unanswerable else 1.0
    )

    # Fact coverage (answerable only)
    avg_fact_coverage = (
        sum(r["fact_coverage"] for r in answerable) / len(answerable)
        if answerable else 0.0
    )

    # Citation validity rate
    all_citations = sum(r["citation_count"] for r in valid)
    all_valid_citations = sum(r["valid_citations"] for r in valid)
    overall_citation_validity = (
        all_valid_citations / all_citations if all_citations else 1.0
    )

    # Source precision
    avg_source_precision = (
        sum(r["source_precision"] for r in answerable) / len(answerable)
        if answerable else 0.0
    )

    # False answer rate on unanswerable
    false_answers = [r for r in unanswerable if r.get("false_answer")]
    false_answer_rate = (
        len(false_answers) / len(unanswerable) if unanswerable else 0.0
    )

    # MRR (Mean Reciprocal Rank) - based on whether expected source appears in citations
    mrr_sum = 0.0
    mrr_count = 0
    for r in answerable:
        expected = set(r.get("expected_sources", []))
        if expected:
            cited = r.get("cited_files", [])
            for rank, f in enumerate(cited, 1):
                if f in expected:
                    mrr_sum += 1.0 / rank
                    break
            mrr_count += 1
    mrr = mrr_sum / mrr_count if mrr_count else 0.0

    return {
        "total_questions": total,
        "errors": len(errors),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "avg_fact_coverage": avg_fact_coverage,
        "overall_citation_validity": overall_citation_validity,
        "avg_source_precision": avg_source_precision,
        "false_answer_rate": false_answer_rate,
        "mrr": mrr,
    }


def print_report(metrics: dict, mode: str):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT ({mode.upper()} MODE)")
    print("=" * 60)
    print(f"  Total questions:         {metrics['total_questions']}")
    print(f"  Errors:                  {metrics['errors']}")
    print(f"  Answerable:              {metrics['answerable_questions']}")
    print(f"  Unanswerable:            {metrics['unanswerable_questions']}")
    print("  ---")
    print(f"  Abstention Precision:    {metrics['abstention_precision']:.2%}")
    print(f"  Abstention Recall:       {metrics['abstention_recall']:.2%}")
    print(f"  Avg Fact Coverage:       {metrics['avg_fact_coverage']:.2%}")
    print(f"  Citation Validity Rate:  {metrics['overall_citation_validity']:.2%}")
    print(f"  Avg Source Precision:    {metrics['avg_source_precision']:.2%}")
    print(f"  False Answer Rate:       {metrics['false_answer_rate']:.2%}")
    print(f"  Mean Reciprocal Rank:    {metrics['mrr']:.4f}")
    print("=" * 60)

    # Interpretation notes
    print("\n  METRIC DEFINITIONS:")
    print("  * Abstention Precision: Of all responses that abstained, fraction that should have.")
    print("  * Abstention Recall:    Of all unanswerable questions, fraction correctly abstained.")
    print("  * Fact Coverage:        Fraction of expected facts found in answer (keyword match).")
    print("  * Citation Validity:    Fraction of citations with valid chunk_id and quoted_text.")
    print("  * Source Precision:     Fraction of cited files that match expected source files.")
    print("  * False Answer Rate:    Fraction of unanswerable questions answered confidently.")
    print("  * MRR:                  Mean Reciprocal Rank of expected source in citations.")

    if mode == "offline":
        print("\n  NOTE: Offline mode uses mock generator and deterministic embeddings.")
        print("  Fact coverage and source precision reflect pipeline wiring, not semantic quality.")
        print("  Run with --mode online and a valid MISTRAL_API_KEY for real LLM evaluation.")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--mode", choices=["offline", "local", "online"], default="offline",
        help="Evaluation mode: offline (mock/deterministic), local (sentence-transformers/mock), or online (real Mistral)",
    )
    parser.add_argument(
        "--questions", default="evaluation/questions.json",
        help="Path to evaluation questions JSON",
    )
    parser.add_argument(
        "--data-dir", default="data/sample_documents",
        help="Path to sample documents directory",
    )
    args = parser.parse_args()

    # Configure environment based on mode
    if args.mode == "offline":
        os.environ["USE_DETERMINISTIC_EMBEDDER"] = "true"
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["MIN_RELEVANCE_SCORE"] = "50.0"  # Lenient for deterministic embeddings
    elif args.mode == "local":
        os.environ["USE_DETERMINISTIC_EMBEDDER"] = "false"
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["MIN_RELEVANCE_SCORE"] = "1.2"
    else:
        if not os.environ.get("MISTRAL_API_KEY"):
            print("ERROR: --mode online requires MISTRAL_API_KEY to be set.")
            sys.exit(1)
        os.environ["USE_DETERMINISTIC_EMBEDDER"] = "false"
        os.environ["LLM_PROVIDER"] = "mistral"

    # Use a temporary chroma dir for evaluation
    import tempfile
    eval_dir = tempfile.mkdtemp(prefix="rag_eval_")
    os.environ["CHROMA_PERSIST_DIRECTORY"] = os.path.join(eval_dir, "chroma")
    os.environ["CHROMA_COLLECTION_NAME"] = "evaluation_collection"

    # Clear cached settings
    from app.core.config import get_settings
    get_settings.cache_clear()

    print(f"Running evaluation in {args.mode} mode...")
    print(f"Questions: {args.questions}")
    print(f"Data dir: {args.data_dir}")

    # Load questions
    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} evaluation questions.")

    # Create test client
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        # Ingest sample documents
        print("\nIngesting sample documents...")
        doc_ids = ingest_sample_documents(client, args.data_dir)
        if not doc_ids:
            print("WARNING: No documents ingested. Results will reflect empty corpus.")

        # Run evaluation
        print(f"\nRunning {len(questions)} evaluation questions...")
        results = run_evaluation(questions, client, doc_ids)

    # Compute and print metrics
    metrics = compute_aggregate_metrics(results)
    print_report(metrics, args.mode)

    # Save detailed results
    output_path = Path("evaluation") / f"results_{args.mode}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
