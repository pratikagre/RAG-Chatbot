import os
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly or as a module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    Loads the PDF file from the given path, splits into standard overlapping chunks,
    and enriches each chunk with structured metadata (page number, source, chunk ID).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    logger.info(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()
    logger.info(f"Successfully loaded {len(raw_pages)} pages from {pdf_path}.")

    # Configure recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    raw_chunks = text_splitter.split_documents(raw_pages)
    logger.info(f"Created {len(raw_chunks)} chunks from {len(raw_pages)} pages.")

    # Enrich metadata
    processed_chunks: List[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        # 1-indexed page number
        page_num = chunk.metadata.get("page", 0) + 1
        enriched_metadata = {
            "chunk_id": f"chunk_{idx+1}",
            "source": Path(pdf_path).name,
            "page": page_num,
            "char_count": len(chunk.page_content)
        }
        # Strip extraneous whitespace
        clean_text = chunk.page_content.strip()
        if clean_text:
            processed_chunks.append(Document(page_content=clean_text, metadata=enriched_metadata))

    logger.info(f"Total valid non-empty chunks ready for indexing: {len(processed_chunks)}")
    return processed_chunks


def run_ingestion(pdf_path: str = settings.PDF_PATH, index_name: str = settings.PINECONE_INDEX_NAME):
    """
    Full ETL pipeline:
    1. Parse and chunk the eBook PDF.
    2. Initialize Pinecone index (if not already existing).
    3. Generate OpenAI embeddings and upsert chunks into Pinecone.
    """
    logger.info("================ Starting Document Ingestion Pipeline ================")
    chunks = load_and_split_pdf(pdf_path)

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        logger.warning(
            "OPENAI_API_KEY is not set or using placeholder! "
            "Please configure your OpenAI and Pinecone API keys in .env to upsert to Pinecone."
        )
        return chunks

    if not settings.PINECONE_API_KEY or settings.PINECONE_API_KEY == "your_pinecone_api_key_here":
        logger.warning(
            "PINECONE_API_KEY is not set or using placeholder! "
            "Please configure your Pinecone API key in .env to upsert to Pinecone."
        )
        return chunks

    # Initialize Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Creating serverless Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION
            )
        )
        logger.info(f"Index '{index_name}' successfully created.")
    else:
        logger.info(f"Index '{index_name}' found in Pinecone.")

    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )

    logger.info(f"Upserting {len(chunks)} document chunks to Pinecone index '{index_name}'...")
    
    # Upsert using LangChain's PineconeVectorStore wrapper
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    logger.info("================ Ingestion Pipeline Successfully Completed! ================")
    return vector_store


if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else settings.PDF_PATH
    idx = sys.argv[2] if len(sys.argv) > 2 else settings.PINECONE_INDEX_NAME
    run_ingestion(pdf_file, idx)
