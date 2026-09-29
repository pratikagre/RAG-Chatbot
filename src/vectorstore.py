import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import settings

logger = logging.getLogger(__name__)

class LocalFallbackVectorStore:
    """
    In-memory fallback vector store when Pinecone or OpenAI credentials 
    are not yet configured. Enables offline testing and development.
    Uses TF-IDF / BM25-style keyword and n-gram overlap scoring.
    """
    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents: List[Document] = documents or []

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []

        import re
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return [(doc, 0.0) for doc in self.documents[:k]]

        scored_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            doc_words = set(re.findall(r"\w+", content_lower))
            overlap = query_words.intersection(doc_words)
            # Calculate Jaccard / lexical similarity score
            score = len(overlap) / (len(query_words) + 1e-5)
            scored_docs.append((doc, float(min(1.0, score))))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        k = (search_kwargs or {}).get("k", 4)
        class LocalRetriever:
            def __init__(self, store, k):
                self.store = store
                self.k = k
            def invoke(self, query: str) -> List[Document]:
                results = self.store.similarity_search_with_relevance_scores(query, k=self.k)
                return [doc for doc, score in results]
        return LocalRetriever(self, k)


_local_store_cache: Optional[LocalFallbackVectorStore] = None


def get_embeddings():
    """Initializes embeddings with fallback if API key is not present."""
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here":
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
    return None


def get_pinecone_client() -> Optional[Pinecone]:
    """Initializes Pinecone client if API key is present."""
    if settings.PINECONE_API_KEY and settings.PINECONE_API_KEY != "your_pinecone_api_key_here":
        return Pinecone(api_key=settings.PINECONE_API_KEY)
    return None


def ensure_pinecone_index(pc: Pinecone, index_name: str) -> None:
    """Ensures that the Pinecone index exists, creating it if needed."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index '{index_name}' (dimension={settings.EMBEDDING_DIMENSION}, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION
            )
        )
        logger.info(f"Pinecone index '{index_name}' successfully created.")
    else:
        logger.info(f"Pinecone index '{index_name}' already exists.")


def get_vector_store():
    """
    Returns the configured Pinecone VectorStore.
    Falls back gracefully to LocalFallbackVectorStore if credentials are not set.
    """
    global _local_store_cache
    pc = get_pinecone_client()
    embeddings = get_embeddings()

    if pc and embeddings:
        try:
            ensure_pinecone_index(pc, settings.PINECONE_INDEX_NAME)
            logger.info(f"Connecting to Pinecone index '{settings.PINECONE_INDEX_NAME}'...")
            return PineconeVectorStore(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=embeddings
            )
        except Exception as e:
            logger.warning(f"Error initializing Pinecone index: {e}. Falling back to local store.")

    # Fallback to local store
    if _local_store_cache is None:
        from src.ingestion import load_and_split_pdf
        chunks = load_and_split_pdf(settings.PDF_PATH)
        _local_store_cache = LocalFallbackVectorStore(chunks)
    return _local_store_cache
