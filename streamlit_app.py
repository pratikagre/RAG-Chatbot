import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.config import settings
from src.graph import run_rag_pipeline
from src.ingestion import run_ingestion

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .score-badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .score-badge-medium {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .score-badge-refusal {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .chunk-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=70)
    st.title("System Control")
    st.markdown("**Agentic AI RAG System**")
    
    st.markdown("---")
    st.subheader("⚙️ Configuration")
    st.text(f"Vector DB: Pinecone")
    st.text(f"Index: {settings.PINECONE_INDEX_NAME}")
    st.text(f"Embedding: {settings.EMBEDDING_MODEL}")
    st.text(f"LLM: {settings.LLM_MODEL}")
    st.text(f"Top-K: {settings.TOP_K}")
    
    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    pdf_exists = os.path.exists(settings.PDF_PATH)
    if pdf_exists:
        st.success(f"PDF Found: `{Path(settings.PDF_PATH).name}` (60 pages)")
    else:
        st.error("PDF not found in data directory.")
        
    if st.button("🔄 Run Document Ingestion"):
        with st.spinner("Parsing PDF and indexing into Pinecone..."):
            try:
                run_ingestion()
                st.success("Ingestion completed successfully!")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    st.markdown("---")
    st.subheader("🎯 Benchmark Test Queries")
    st.caption("Click any query below to test grounded retrieval & refusal:")
    
    sample_queries = [
        ("1. Definition & Scope", "What is the core definition of Agentic AI as outlined in the eBook?"),
        ("2. Architecture & Paradigms", "What are the main architectural components required to build agentic systems?"),
        ("3. Industry Use Cases", "What real-world industry use cases for Agentic AI are discussed in the eBook?"),
        ("4. Agentic AI vs Traditional", "How does Agentic AI differ from traditional generative AI chatbots according to the text?"),
        ("5. Key Challenges", "What key challenges or limitations of Agentic AI are mentioned in the document?"),
        ("6. Out-of-Scope (Refusal Test)", "What is the capital of France?")
    ]
    
    selected_sample = None
    for label, q_text in sample_queries:
        if st.button(label, key=f"btn_{label}"):
            selected_sample = q_text

# Main View
st.markdown('<div class="main-title">🤖 Agentic AI RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Stateful RAG pipeline built with <b>LangGraph</b>, <b>Pinecone</b>, and strictly grounded response generation.</div>', unsafe_allow_html=True)

# Layout: 2 Columns (Chat on Left, Inspector on Right)
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Interactive Chat")
    
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("score") is not None:
                score = msg["score"]
                if score >= 0.80:
                    st.markdown(f'<span class="score-badge-high">Confidence Score: {score:.2f} (High)</span>', unsafe_allow_html=True)
                elif score > 0.0:
                    st.markdown(f'<span class="score-badge-medium">Confidence Score: {score:.2f} (Moderate)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="score-badge-refusal">Confidence Score: 0.00 (Out-of-Scope / Refusal)</span>', unsafe_allow_html=True)

    # Handle Input (either from chat input or sample query button)
    user_input = st.chat_input("Ask any question about the Agentic AI eBook...") or selected_sample

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run LangGraph pipeline
        with st.chat_message("assistant"):
            with st.spinner("Retrieving from Pinecone & evaluating groundedness..."):
                response_payload = run_rag_pipeline(user_input)
                st.session_state.last_response = response_payload
                
                answer = response_payload["final_answer"]
                score = response_payload["confidence_score"]
                st.markdown(answer)
                
                if score >= 0.80:
                    st.markdown(f'<span class="score-badge-high">Confidence Score: {score:.2f} (High Grounding)</span>', unsafe_allow_html=True)
                elif score > 0.0:
                    st.markdown(f'<span class="score-badge-medium">Confidence Score: {score:.2f} (Moderate Grounding)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="score-badge-refusal">Confidence Score: 0.00 (Out-of-Scope Refusal)</span>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "score": score
        })

with col2:
    st.subheader("🔍 Context & Grounding Inspector")
    
    last = st.session_state.last_response
    if last:
        st.markdown(f"**Current Query:** *{last['query']}*")
        
        # Metric card
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Confidence Score", f"{last['confidence_score']:.2f}")
        with metric_col2:
            st.metric("Chunks Retrieved", len(last["retrieved_context_chunks"]))
            
        st.markdown("---")
        st.markdown("#### 📑 Retrieved Context Chunks")
        metadata_list = last.get("metadata", [])
        
        for i, chunk in enumerate(last["retrieved_context_chunks"]):
            meta = metadata_list[i] if i < len(metadata_list) else {}
            page = meta.get("page", "N/A")
            chunk_id = meta.get("chunk_id", f"chunk_{i+1}")
            relevance = meta.get("relevance_score", "N/A")
            
            with st.expander(f"Chunk {i+1} | Page {page} | Score: {relevance}"):
                st.markdown(f"```text\n{chunk}\n```")

        st.markdown("---")
        st.markdown("#### 📦 Standard Output Payload (JSON)")
        payload_view = {
            "query": last["query"],
            "final_answer": last["final_answer"],
            "retrieved_context_chunks": last["retrieved_context_chunks"],
            "confidence_score": last["confidence_score"]
        }
        st.json(payload_view)
    else:
        st.info("Submit a question or click a benchmark query from the sidebar to inspect retrieved chunks, relevance metrics, and JSON output payload.")
