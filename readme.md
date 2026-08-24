# (RAG PIPELINE) System using Gemini, Chroma DB, and without LangChain

A lightweight, high-performance, production-ready **Retrieval-Augmented Generation (RAG)** system built strictly using **Google Gemini 3.6 Flash**, **Chroma DB**, and **Python** — built entirely **without LangChain** or third-party orchestration frameworks.

---

## 📌 Brief Overview

Many modern RAG applications rely heavily on framework abstractions like LangChain or LlamaIndex, which can introduce hidden complexity, latency overhead, and debugging friction. 

This project demonstrates how to build a **fully functional, robust RAG Pipeline from scratch** using native Python code, direct vector similarity search via **Chroma DB**, and direct LLM generation/tool calling using **Google Gemini**.

### 🌟 Key Features
- **Zero Framework Dependency**: No LangChain / LlamaIndex overhead. Pure, transparent Python logic.
- **Custom Paragraph Chunking**: Intelligently splits document text while retaining section metadata.
- **Persistent Vector Store**: Uses **Chroma DB** to store and query vector embeddings persistently on disk.
- **Grounded LLM Generation**: Uses `gemini-3.6-flash` with strict system prompt grounding to eliminate hallucinations.
- **Context Refusal Fallback**: Explicitly returns *"I don't have enough context or information to answer this question"* for out-of-domain queries.
- **Native Agentic Tool Calling**: Uses Gemini's native function calling (`search_docs`) to allow the model to dynamically choose when to query internal documents versus answering direct general knowledge questions.
- **Automated Evaluation Suite**: Benchmark script (`evaluate.py`) to verify accuracy, keyword coverage, and refusal performance.

---

## 🔄 End-to-End Process Workflow

```mermaid
flowchart TD
    A[📄 Raw Documents\n.txt files in data/] --> B[🔪 Semantic Chunking\nParagraph splitter & metadata tagging]
    B --> C[🧠 Vector Embedding & Storage\nChroma DB Persistent Collection]
    
    subgraph User Interaction & Retrieval
        D[❓ User Question] --> E{Select Mode}
        E -->|Standard RAG| F[🔍 Chroma DB Vector Query\nTop-K Relevant Chunks]
        F --> G[📝 Context Assembly & Grounded Prompt]
        G --> H[🤖 Gemini 3.6 Flash Generation]
        
        E -->|Agentic RAG| I[🛠️ Gemini Chat with Function Calling]
        I -->|Triggers search_docs| F
    end

    H --> J[💡 Fact-Grounded Response]
    I -->|Direct Answer / Tool Result| J
```

### Detailed Workflow Steps:

1. **Step 1: Document Ingestion & Storage (`data/`)**
   - Text documents (HR Policy, Engineering Standards, Onboarding Guide, Product Knowledge Base, Security Policy) are stored in the `data/` folder.

2. **Step 2: Semantic Paragraph Chunking (`ingest.py`)**
   - Documents are split by paragraph boundaries (`\n\n`), ignoring short header lines and formatting noise. Each chunk is tagged with its origin metadata (e.g., `{"source": "HR Policy"}`).

3. **Step 3: Embedding & Vector Indexing (`Chroma DB`)**
   - The document chunks, IDs, and metadata are indexed into a persistent ChromaDB collection stored locally at `./chroma_db`.

4. **Step 4: Vector Similarity Search & Context Retrieval (`query.py`)**
   - User questions are queried against ChromaDB using cosine similarity to retrieve the top $K$ ($N=3$) most relevant context chunks.

5. **Step 5: Grounded LLM Response Generation (`gemini-3.6-flash`)**
   - The retrieved context chunks and user question are injected into a strict system prompt instructing Gemini to answer **only** based on the provided context, refusing to hallucinate if the answer is missing.

6. **Step 6: Native Agentic Function Tooling (`search_docs`)**
   - In Agentic Mode, Gemini is provided with a native function tool `search_docs`. Gemini dynamically decides whether it needs to call `search_docs` for internal company information or answer directly for general math/conversational queries.

---

## 📁 Repository Structure

```text
RAG PIPELINE BASIC/
│
├── data/                         # Folder containing knowledge base text files
│   ├── company_hr_policy.txt
│   ├── engineering_standards.txt
│   ├── onboarding_guide.txt
│   ├── product_knowledge_base.txt
│   └── security_policy.txt
│
├── chroma_db/                    # Persistent vector database directory (auto-created)
│
├── .env.example                  # Environment variables template
├── .env                          # Local environment variables file (contains GEMINI_API_KEY)
├── .gitignore                    # Git ignore rules for secrets, venv, and vector DB
│
├── ingest.py                     # Document loader & ChromaDB ingestion pipeline
├── query.py                      # Core retrieval engine, Standard RAG & Agentic RAG logic
├── app.py                        # Interactive CLI query application
├── evaluate.py                   # Automated benchmark evaluation suite
├── requirements.txt              # Project dependencies list
└── RAG_pipeline_from_scratch.ipynb  # Initial step-by-step notebook prototype
```

---

## ⚙️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/akhiranandan-04/Simple-RAG-Pipeline-.git
cd Simple-RAG-Pipeline-
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy `.env.example` to `.env` and set your Google Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🚀 Running the System

### 1. Ingest Documents into Vector Store
```bash
python ingest.py
```

### 2. Run Interactive CLI App
```bash
python app.py
```
You can also run direct queries:
```bash
# Standard Grounded RAG
python app.py -q "What is the work from home policy?"

# Agentic RAG Mode
python app.py -a -q "How many sick leave days do I get per year?"
```

### 3. Run Automated Evaluation Suite
```bash
python evaluate.py
```

---

## 📊 Tech Stack

| Component | Technology |
|---|---|
| **LLM Engine** | Google Gemini (`gemini-3.6-flash`) |
| **Vector Database** | Chroma DB (Persistent Client) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) / Chroma Default |
| **Orchestration** | Pure Python (No LangChain / No LlamaIndex) |
| **Language** | Python 3.10+ |

---

## 📄 License
Distributed under the MIT License.
