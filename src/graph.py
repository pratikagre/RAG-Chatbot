import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.nodes import retrieve_node, generate_node, evaluate_groundedness_node

logger = logging.getLogger(__name__)


def build_rag_graph():
    """
    Constructs and compiles the cyclic/linear LangGraph RAG workflow.
    Flow:
      START -> retrieve -> generate -> evaluate_groundedness -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate_groundedness", evaluate_groundedness_node)

    # 2. Define Workflow Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate_groundedness")
    workflow.add_edge("evaluate_groundedness", END)

    # 3. Compile Graph
    compiled_app = workflow.compile()
    return compiled_app


# Cached compiled graph instance
rag_graph = build_rag_graph()


def run_rag_pipeline(query: str) -> Dict[str, Any]:
    """
    Executes the LangGraph RAG pipeline and returns the standardized output payload:
    {
        "query": str,
        "final_answer": str,
        "retrieved_context_chunks": List[str],
        "confidence_score": float
    }
    """
    initial_state: AgentState = {
        "question": query,
        "context": [],
        "context_metadata": [],
        "final_answer": "",
        "confidence_score": 0.0,
        "grounded": False,
        "is_refusal": False
    }

    result = rag_graph.invoke(initial_state)

    return {
        "query": query,
        "final_answer": result.get("final_answer", ""),
        "retrieved_context_chunks": result.get("context", []),
        "confidence_score": result.get("confidence_score", 0.0),
        "metadata": result.get("context_metadata", [])
    }


if __name__ == "__main__":
    test_query = "What is the core definition of Agentic AI as outlined in the eBook?"
    print(f"Running pipeline with query: '{test_query}'")
    out = run_rag_pipeline(test_query)
    print("\n--- Output ---")
    print(f"Query: {out['query']}")
    print(f"Final Answer:\n{out['final_answer']}")
    print(f"Confidence Score: {out['confidence_score']}")
    print(f"Retrieved Chunks: {len(out['retrieved_context_chunks'])}")
