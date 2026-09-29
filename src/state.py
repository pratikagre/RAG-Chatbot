from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Typed state dictionary representing the workflow state in LangGraph.
    Tracks query, retrieved context, generated answer, and evaluation metrics.
    """
    question: str
    context: List[str]
    context_metadata: List[Dict[str, Any]]
    final_answer: str
    confidence_score: float
    grounded: bool
    is_refusal: bool
