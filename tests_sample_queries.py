import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import run_rag_pipeline

SAMPLE_QUERIES = [
    {
        "category": "Definition & Scope",
        "query": "What is the core definition of Agentic AI as outlined in the eBook?",
        "expected_grounded": True,
        "expect_refusal": False
    },
    {
        "category": "Architecture & Paradigms",
        "query": "What are the main architectural components required to build agentic systems?",
        "expected_grounded": True,
        "expect_refusal": False
    },
    {
        "category": "Use Cases",
        "query": "What real-world industry use cases for Agentic AI are discussed in the eBook?",
        "expected_grounded": True,
        "expect_refusal": False
    },
    {
        "category": "Comparison",
        "query": "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "expected_grounded": True,
        "expect_refusal": False
    },
    {
        "category": "Challenges & Considerations",
        "query": "What key challenges or limitations of Agentic AI are mentioned in the document?",
        "expected_grounded": True,
        "expect_refusal": False
    },
    {
        "category": "Out-of-Scope (Groundedness Check)",
        "query": "What is the capital of France?",
        "expected_grounded": False,
        "expect_refusal": True
    }
]


def test_payload_structure(result: dict, query_str: str):
    """Validates that output matches the strict JSON schema expected by the assignment."""
    assert "query" in result, "Missing 'query' field in result payload"
    assert result["query"] == query_str, "Query in payload does not match input"
    assert "final_answer" in result, "Missing 'final_answer' field in result payload"
    assert isinstance(result["final_answer"], str) and len(result["final_answer"]) > 0, "final_answer must be non-empty string"
    assert "retrieved_context_chunks" in result, "Missing 'retrieved_context_chunks' in result payload"
    assert isinstance(result["retrieved_context_chunks"], list), "retrieved_context_chunks must be a list"
    assert "confidence_score" in result, "Missing 'confidence_score' in result payload"
    assert isinstance(result["confidence_score"], (int, float)), "confidence_score must be a number"
    assert 0.0 <= result["confidence_score"] <= 1.0, "confidence_score must be between 0.0 and 1.0"


def run_benchmark_tests():
    print("=" * 80)
    print("      RUNNING BENCHMARK EVALUATION ON 6 SAMPLE QUERIES (LangGraph RAG)")
    print("=" * 80)
    
    passed_count = 0
    results_summary = []

    for idx, test_case in enumerate(SAMPLE_QUERIES, 1):
        cat = test_case["category"]
        query = test_case["query"]
        expect_refusal = test_case["expect_refusal"]
        
        print(f"\n[{idx}/6] Category: {cat}")
        print(f"      Query: \"{query}\"")
        
        result = run_rag_pipeline(query)
        test_payload_structure(result, query)
        
        answer = result["final_answer"]
        score = result["confidence_score"]
        chunks = result["retrieved_context_chunks"]

        refusal_markers = [
            "cannot answer", "not covered in the document", 
            "not available", "does not contain"
        ]
        is_refusal = any(m in answer.lower() for m in refusal_markers)

        if expect_refusal:
            # Out of scope query MUST refuse and yield 0.0 confidence
            is_valid = is_refusal and score == 0.0
            status_str = "PASS (Correctly Refused Out-of-Scope Query)" if is_valid else "FAIL (Did not refuse)"
        else:
            # In scope queries MUST be answered and have confidence > 0
            is_valid = (not is_refusal) and (score > 0.5) and len(chunks) > 0
            status_str = "PASS (Grounded In-Scope Answer)" if is_valid else "FAIL (Failed grounding)"

        if is_valid:
            passed_count += 1

        print(f"      Status: {status_str}")
        print(f"      Confidence Score: {score}")
        print(f"      Context Chunks Retrieved: {len(chunks)}")
        preview = answer.replace("\n", " ")[:140] + ("..." if len(answer) > 140 else "")
        print(f"      Answer Preview: {preview}")

        results_summary.append({
            "index": idx,
            "category": cat,
            "query": query,
            "score": score,
            "status": "PASS" if is_valid else "FAIL",
            "preview": preview
        })

    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed_count}/{len(SAMPLE_QUERIES)} Benchmark Queries Passed.")
    print("=" * 80)

    if passed_count == len(SAMPLE_QUERIES):
        print(">> ALL BENCHMARK VERIFICATION TESTS PASSED SUCCESSFULLY! <<\n")
        return 0
    else:
        print(">> SOME BENCHMARK TESTS FAILED! <<\n")
        return 1


if __name__ == "__main__":
    exit_code = run_benchmark_tests()
    sys.exit(exit_code)
