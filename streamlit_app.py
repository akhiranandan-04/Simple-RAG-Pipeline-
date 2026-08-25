"""
NovaTech RAG Chatbot — Streamlit UI (Gemini + ChromaDB version)
===============================================================

"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from ingest import run_ingestion
from query import get_chroma_collection, get_gemini_client, retrieve, ask_rag, rag_agent

# ──────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────

st.set_page_config(
    page_title="NovaTech Assistant",
    page_icon="💬",
    layout="centered"
)

st.title("NovaTech Internal Assistant")
st.caption("Ask anything about company policies, products, or procedures.")
st.divider()

# ──────────────────────────────────────
# RAG SETUP
# ──────────────────────────────────────

@st.cache_resource
def load_rag():
    """
    Initialize ChromaDB collection and Gemini client.
    """
    collection = get_chroma_collection()
    client = get_gemini_client()
    count = collection.count()
    return collection, client, count


with st.spinner("Loading and indexing company documents..."):
    collection, gemini_client, total_chunks = load_rag()

# Sidebar options
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Select Pipeline Mode:",
        ["Standard Grounded RAG", "Agentic RAG (Tool Calling)"],
        index=0
    )
    st.divider()
    st.markdown(f"**Total Indexed Chunks:** `{total_chunks}`")
    
    if st.button("🔄 Re-run Document Ingestion"):
        with st.spinner("Re-ingesting documents..."):
            run_ingestion()
            st.cache_resource.clear()
            st.rerun()

st.success(f"Ready — {total_chunks} document chunks indexed.", icon="✅")
st.divider()

# ──────────────────────────────────────
# CHAT HISTORY
# ──────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────
# RAG FUNCTION
# ──────────────────────────────────────

def execute_rag(question: str, use_agent: bool = False) -> dict:
    """
    Execute the full RAG pipeline: retrieve → augment → generate using Gemini LLM.
    """
    if use_agent:
        answer = rag_agent(question, verbose=False)
        try:
            chunks, sources_meta = retrieve(question, n_results=3)
            source_names = [m.get("source", "Unknown") if isinstance(m, dict) else str(m) for m in sources_meta]
            chunk_pairs = list(zip(chunks, source_names))
            dedup_sources = list(set(source_names))
        except Exception:
            dedup_sources = ["Agent Search Tool"]
            chunk_pairs = []
        return {
            "answer": answer,
            "sources": dedup_sources,
            "chunks": chunk_pairs
        }
    else:
        chunks, sources_meta = retrieve(question, n_results=3)
        source_names = [m.get("source", "Unknown") if isinstance(m, dict) else str(m) for m in sources_meta]
        answer = ask_rag(question, n_results=3, verbose=False)
        return {
            "answer": answer,
            "sources": list(set(source_names)),
            "chunks": list(zip(chunks, source_names))
        }

# ──────────────────────────────────────
# RENDER CHAT HISTORY
# ──────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")
            if msg.get("chunks"):
                with st.expander("View retrieved document chunks"):
                    for i, (chunk_text, source_file) in enumerate(msg["chunks"], 1):
                        st.markdown(f"**Chunk {i} — `{source_file}`**")
                        st.info(chunk_text)

# ──────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────

question = st.chat_input("Ask a question about NovaTech policies or products...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    use_agent = (mode == "Agentic RAG (Tool Calling)")

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = execute_rag(question, use_agent=use_agent)

        st.markdown(result["answer"])
        if result["sources"]:
            st.caption(f"Sources: {', '.join(result['sources'])}")

        if result["chunks"]:
            with st.expander("View retrieved document chunks"):
                for i, (chunk_text, source_file) in enumerate(result["chunks"], 1):
                    st.markdown(f"**Chunk {i} — `{source_file}`**")
                    st.info(chunk_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks": result["chunks"]
    })
