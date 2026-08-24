import os
import sys
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables
load_dotenv()

_chroma_collection = None
_gemini_client = None


def get_gemini_client():
    """
    Get or initialize Google Gemini API client.
    """
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def get_chroma_collection(db_path: str = "./chroma_db", collection_name: str = "company_docs"):
    """
    Get or initialize persistent ChromaDB collection.
    If collection does not exist, run ingestion automatically.
    """
    global _chroma_collection
    if _chroma_collection is None:
        chroma_client = chromadb.PersistentClient(path=db_path)
        try:
            _chroma_collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            print("[!] Collection not found. Running ingestion first...")
            from ingest import run_ingestion
            run_ingestion(db_path=db_path, collection_name=collection_name)
            _chroma_collection = chroma_client.get_collection(name=collection_name)
    return _chroma_collection


def retrieve(question: str, n_results: int = 3):
    """
    Retrieve top n_results relevant chunks for a question from ChromaDB.
    """
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0], results["metadatas"][0]


def ask_rag(question: str, n_results: int = 3, verbose: bool = True) -> str:
    """
    Standard grounded RAG query function using Gemini LLM.
    """
    chunks, sources = retrieve(question, n_results)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[?] Question: {question}")
        print(f"{'-' * 60}")
        print(f"[*] Retrieved {len(chunks)} chunks:")
        for chunk, source in zip(chunks, sources):
            print(f"   [{source['source']}] {chunk[:80]}...")
        print(f"{'-' * 60}")

    context = "\n\n".join(chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the provided context.\n"
        "If the context does not contain information to answer this question, "
        "say 'I don't have enough context or information to answer this question.'\n"
        "Do not make up information or assume anything. Strictly answer only from the provided context."
    )

    user_prompt = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )

    answer = response.text

    if verbose:
        print(f"[>] Answer: {answer}")
        print(f"{'=' * 60}\n")

    return answer


def search_docs(query: str) -> str:
    """
    Tool function used by Agentic RAG to search company documents.
    """
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    chunks = results["documents"][0]
    return "\n\n".join(chunks)


def rag_agent(question: str, verbose: bool = True) -> str:
    """
    Agentic RAG function using Gemini Chat with Function Calling tool.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[?] Question: {question}")
        print(f"{'-' * 60}")

    system_instruction = (
        "You are a helpful company assistant with access to internal documents "
        "including HR policies, Product, Engineering, Onboarding, and Security.\n"
        "Use the search_docs tool to find answers from company documents.\n"
        "If the documents do not contain the answer, say so clearly.\n"
        "Always base your answers on the retrieved documents if you decide to use the tool.\n"
        "If tool use is not needed, specify that you are not using the tool and answer based on your knowledge."
    )

    client = get_gemini_client()
    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[search_docs],
            temperature=0.2,
        ),
    )

    response = chat.send_message(question)
    answer = response.text

    if verbose:
        print(f"[>] Answer: {answer}")
        print(f"{'=' * 60}\n")

    return answer


if __name__ == "__main__":
    print("Testing query module...")
    ask_rag("What is the work from home policy?", n_results=3)
