"""
NovaTech RAG Chatbot — Streamlit UI (TF-IDF Lightweight Deployment Version)
=============================================================================

PURPOSE
-------
This is the lightweight production/deployment version of the chatbot UI.
It replaces ChromaDB with scikit-learn TF-IDF vector search — an ultra-lightweight
alternative that works seamlessly on free cloud deployment tiers (e.g., Render, Railway).

WHY USE THIS FILE FOR FREE DEPLOYMENT
--------------------------------------
ChromaDB automatically downloads the all-MiniLM-L6-v2 ONNX model (~79MB) on first start
and requires ~300MB+ RAM. On free cloud tiers with strict RAM limits (e.g., 512MB total RAM),
this can trigger out-of-memory (OOM) crashes.

TF-IDF uses 0 external model downloads, consumes ~5MB RAM, and starts in under 2 seconds.

HOW TO RUN LOCALLY
------------------
    streamlit run streamlit_app_tfidf.py
    Open: http://localhost:8501
"""

import os
import sys
import glob
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ──────────────────────────────────────
# CONFIG & PATHS
# ──────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.dirname(__file__)

# ──────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────

st.set_page_config(
    page_title="NovaTech Assistant (TF-IDF)",
    page_icon="💬",
    layout="centered"
)

st.title("NovaTech Internal Assistant (Lightweight TF-IDF)")
st.caption("Ask anything about company policies, products, or procedures — Powered by TF-IDF + Gemini.")
st.divider()

# ──────────────────────────────────────
# RAG SETUP — TF-IDF Indexing
# ──────────────────────────────────────

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Gemini API key not found. Please set GEMINI_API_KEY in your environment or .env file.")
        st.stop()
    return genai.Client(api_key=api_key)


@st.cache_resource
def load_tfidf_rag():
    """
    Read text documents, chunk by paragraph, and build a TF-IDF index.
    Cached so it runs only ONCE per application session.
    """
    chunks = []
    sources = []

    file_source_map = {
        "company_hr_policy.txt": "HR Policy",
        "engineering_standards.txt": "Engineering",
        "onboarding_guide.txt": "Onboarding",
        "product_knowledge_base.txt": "Product",
        "security_policy.txt": "Security"
    }

    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not txt_files:
        st.error(f"No .txt documents found in '{DATA_DIR}' directory.")
        st.stop()

    for file_path in sorted(txt_files):
        filename = os.path.basename(file_path)
        source_name = file_source_map.get(filename, filename.replace(".txt", "").replace("_", " ").title())

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        for para in text.strip().split("\n\n"):
            para = para.strip()
            if len(para) < 50:
                continue
            if para.startswith("===="):
                continue

            chunks.append(para)
            sources.append(source_name)

    # Build TF-IDF document-term matrix
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    client = get_gemini_client()

    return chunks, sources, vectorizer, tfidf_matrix, client


with st.spinner("Loading documents and building TF-IDF index..."):
    chunks, sources, vectorizer, tfidf_matrix, gemini_client = load_tfidf_rag()

with st.sidebar:
    st.header("⚡ Deployment Settings")
    st.info("💡 **TF-IDF Mode**: Ultra-lightweight (~5MB RAM). Recommended for free Cloud / Render deployment.")
    st.divider()
    st.markdown(f"**Total Indexed Chunks:** `{len(chunks)}`")
    if st.button("🔄 Clear Cache & Reload Index"):
        st.cache_resource.clear()
        st.rerun()

st.success(f"Ready — {len(chunks)} document chunks indexed using lightweight TF-IDF search.", icon="⚡")
st.divider()

# ──────────────────────────────────────
# CHAT HISTORY
# ──────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────
# RAG FUNCTION — TF-IDF Retrieval
# ──────────────────────────────────────

def ask_tfidf_rag(question: str, top_k: int = 3) -> dict:
    """
    Execute full RAG pipeline: TF-IDF vector retrieval → Augment → Gemini LLM generation.
    """
    # Step 1: Transform question to TF-IDF vector
    question_vec = vectorizer.transform([question])

    # Step 2: Compute cosine similarity against all document chunk vectors
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()

    # Step 3: Select top_k indices sorted descending
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_chunks = [chunks[i] for i in top_indices]
    retrieved_sources = [sources[i] for i in top_indices]
    retrieved_scores = [float(similarities[i]) for i in top_indices]

    context = "\n\n---\n\n".join(retrieved_chunks)

    system_prompt = (
        "You are a helpful NovaTech company assistant.\n"
        "Answer questions using ONLY the provided context.\n"
        "If the context does not contain enough information to answer, "
        "say 'I don't have enough context or information to answer this question.'\n"
        "Do not make up information. Be concise and direct."
    )

    user_prompt = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )

    answer = response.text

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(retrieved_sources)),
        "chunks": list(zip(retrieved_chunks, retrieved_sources, retrieved_scores))
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
                    for i, (chunk_text, source_file, score) in enumerate(msg["chunks"], 1):
                        st.markdown(f"**Chunk {i} — `{source_file}`** (similarity: {score:.3f})")
                        st.info(chunk_text)

# ──────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────

question = st.chat_input("Ask a question about NovaTech policies or products...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents (TF-IDF) and generating answer..."):
            result = ask_tfidf_rag(question, top_k=3)

        st.markdown(result["answer"])
        if result["sources"]:
            st.caption(f"Sources: {', '.join(result['sources'])}")

        if result["chunks"]:
            with st.expander("View retrieved document chunks"):
                for i, (chunk_text, source_file, score) in enumerate(result["chunks"], 1):
                    st.markdown(f"**Chunk {i} — `{source_file}`** (similarity: {score:.3f})")
                    st.info(chunk_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks": result["chunks"]
    })
