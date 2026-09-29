import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import run_rag_pipeline
from src.ingestion import run_ingestion

app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    description="Production-grade Retrieval-Augmented Generation (RAG) system built with LangGraph, Pinecone, and strictly grounded responses.",
    version="1.0.0"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        example="What is the core definition of Agentic AI as outlined in the eBook?",
        description="The user question to be answered strictly based on the eBook."
    )


class QueryResponse(BaseModel):
    query: str = Field(..., description="The original query submitted.")
    final_answer: str = Field(..., description="Synthesized grounded answer or refusal.")
    retrieved_context_chunks: List[str] = Field(..., description="List of raw text chunks retrieved from Pinecone/vector store.")
    confidence_score: float = Field(..., description="Confidence / groundedness score between 0.0 and 1.0.")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is Agentic AI?",
                "final_answer": "Agentic AI refers to autonomous systems that understand goals, anticipate needs, and take proactive actions...",
                "retrieved_context_chunks": [
                    "Imagine Sarah, a busy entrepreneur juggling multiple projects. She's not just using a tool anymore - she's working with an AI assistant that doesn't just follow commands, but understands her goals...",
                    "Agentic AI and LLMs work together to boost performance by combining specialized agents with LLM capabilities..."
                ],
                "confidence_score": 0.92
            }
        }


class IngestResponse(BaseModel):
    status: str
    message: str
    index_name: str
    pdf_path: str


@app.get("/", tags=["General"])
async def root():
    return {
        "title": "Agentic AI RAG Chatbot API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "online"
    }


@app.get("/health", tags=["General"])
async def health_check():
    return {
        "status": "healthy",
        "index_name": settings.PINECONE_INDEX_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "pdf_available": os.path.exists(settings.PDF_PATH)
    }


@app.post("/chat", response_model=QueryResponse, tags=["RAG"])
async def chat_endpoint(request: QueryRequest):
    """
    RAG chat endpoint:
    - Queries Pinecone for top-k relevant chunks from the Agentic AI eBook.
    - Synthesizes a strictly grounded answer using LangGraph.
    - Evaluates groundedness and calculates confidence score.
    """
    try:
        result = run_rag_pipeline(request.query)
        return QueryResponse(
            query=result["query"],
            final_answer=result["final_answer"],
            retrieved_context_chunks=result["retrieved_context_chunks"],
            confidence_score=result["confidence_score"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_endpoint():
    """
    Trigger ingestion of the eBook PDF and upsert vectors into Pinecone.
    """
    try:
        run_ingestion(settings.PDF_PATH, settings.PINECONE_INDEX_NAME)
        return IngestResponse(
            status="success",
            message="PDF successfully parsed, chunked, embedded, and indexed.",
            index_name=settings.PINECONE_INDEX_NAME,
            pdf_path=settings.PDF_PATH
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=True)
