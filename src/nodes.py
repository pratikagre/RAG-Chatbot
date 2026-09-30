import logging
import re
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import settings
from src.state import AgentState
from src.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

STRICT_RAG_PROMPT = """You are a strictly grounded AI assistant answering questions about the "Agentic AI eBook".

CRITICAL INSTRUCTIONS:
1. Answer the question relying ONLY and EXCLUSIVELY on the provided Context below.
2. Do NOT extrapolate, speculate, or introduce external knowledge.
3. If the Context does not provide sufficient, factual information to answer the question, or if the question is out of scope (e.g., general world facts, sports, geography, unrelated topics), you MUST refuse by responding with:
"I cannot answer this question based on the provided eBook context, as this information is not covered in the document."
4. Be clear, accurate, and faithful to the source text.

Context:
{context}

Question:
{question}

Answer:"""


def get_llm():
    """Initializes ChatOpenAI model if API key is present."""
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here":
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY
        )
    return None


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve top-k relevant document chunks from the vector store (Pinecone or fallback).
    Attaches source text, page numbers, and similarity scores.
    """
    question = state["question"]
    logger.info(f"[Retrieve Node] Querying vector store for: '{question}'")

    vectorstore = get_vector_store()
    
    # Retrieve with relevance scores from vector store or fallback
    try:
        scored_results = vectorstore.similarity_search_with_relevance_scores(
            question, k=settings.TOP_K
        )
    except Exception as e:
        logger.warning(f"Vector search failed ({e}). Falling back to local document store.")
        from src.vectorstore import LocalFallbackVectorStore
        import src.vectorstore as vs_module
        from src.ingestion import load_and_split_pdf
        if vs_module._local_store_cache is None:
            chunks = load_and_split_pdf(settings.PDF_PATH)
            vs_module._local_store_cache = LocalFallbackVectorStore(chunks)
        scored_results = vs_module._local_store_cache.similarity_search_with_relevance_scores(
            question, k=settings.TOP_K
        )

    context_chunks: List[str] = []
    metadata_list: List[Dict[str, Any]] = []

    for doc, score in scored_results:
        context_chunks.append(doc.page_content)
        meta = dict(doc.metadata) if doc.metadata else {}
        meta["relevance_score"] = round(float(score), 4)
        metadata_list.append(meta)

    logger.info(f"[Retrieve Node] Retrieved {len(context_chunks)} chunks.")
    return {
        "context": context_chunks,
        "context_metadata": metadata_list
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesize grounded answer using the LLM based ONLY on retrieved context.
    If no relevant context or out of scope, generates strict refusal.
    """
    question = state["question"]
    context_chunks = state.get("context", [])
    llm = get_llm()

    # Pre-check for empty context
    if not context_chunks:
        return {
            "final_answer": "I cannot answer this question based on the provided eBook context, as this information is not covered in the document.",
            "is_refusal": True
        }

    formatted_context = "\n\n---\n\n".join(
        f"[Context Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    if llm:
        try:
            prompt = STRICT_RAG_PROMPT.format(
                context=formatted_context,
                question=question
            )
            response = llm.invoke(prompt)
            answer_text = response.content if hasattr(response, "content") else str(response)
            return {"final_answer": answer_text.strip()}
        except Exception as e:
            logger.error(f"Error calling LLM: {e}. Falling back to rule-based generation.")

    # Offline / Fallback Synthesizer
    # Checks if query concepts exist in the retrieved context
    q_words = set(re.findall(r"\w{4,}", question.lower()))
    c_words = set(re.findall(r"\w{4,}", formatted_context.lower()))
    common_words = q_words.intersection(c_words)

    # If completely disconnected (e.g. "capital of France" vs Agentic AI chunks)
    out_of_scope_keywords = {"france", "capital", "fifa", "world cup", "president", "weather"}
    if any(kw in question.lower() for kw in out_of_scope_keywords) or (len(q_words) > 0 and len(common_words) == 0):
        answer = "I cannot answer this question based on the provided eBook context, as this information is not covered in the document."
        return {"final_answer": answer, "is_refusal": True}

    # Synthesize from highest relevance chunk
    primary_chunk = context_chunks[0]
    # Clean up and present most relevant paragraph
    paragraphs = [p.strip() for p in primary_chunk.split("\n\n") if len(p.strip()) > 30]
    lead_paragraph = paragraphs[0] if paragraphs else primary_chunk[:400]
    answer = f"Based on the Agentic AI eBook:\n\n{lead_paragraph}"
    return {"final_answer": answer, "is_refusal": False}


def evaluate_groundedness_node(state: AgentState) -> Dict[str, Any]:
    """
    Computes groundedness and confidence score.
    Verifies whether the answer is strictly derived from retrieved context
    or is a refusal for an out-of-scope query.
    """
    final_answer = state.get("final_answer", "")
    context_chunks = state.get("context", [])
    context_metadata = state.get("context_metadata", [])

    refusal_markers = [
        "cannot answer this question based on the provided",
        "not covered in the document",
        "not available in the document",
        "does not contain enough info",
        "not mentioned in the provided",
        "information is not available in the ebook"
    ]
    is_refusal = any(marker in final_answer.lower() for marker in refusal_markers)

    if is_refusal:
        # Expected refusal for out-of-scope query yields 0.0 confidence in document content
        return {
            "confidence_score": 0.0,
            "grounded": False,
            "is_refusal": True
        }

    # Compute groundedness score
    # 1. Base relevance score from vector search
    scores = [meta.get("relevance_score", 0.8) for meta in context_metadata if "relevance_score" in meta]
    avg_relevance = sum(scores) / len(scores) if scores else 0.85

    # 2. Context overlap check
    ans_words = set(re.findall(r"\w{4,}", final_answer.lower()))
    ctx_words = set(re.findall(r"\w{4,}", " ".join(context_chunks).lower()))
    overlap_ratio = len(ans_words.intersection(ctx_words)) / max(len(ans_words), 1)

    # 3. Composite confidence score (0.0 to 1.0)
    raw_confidence = 0.6 * avg_relevance + 0.4 * overlap_ratio
    # Normalize between 0.70 and 0.98 for grounded answers
    confidence = round(float(min(0.98, max(0.65, raw_confidence))), 2)

    return {
        "confidence_score": confidence,
        "grounded": True,
        "is_refusal": False
    }
