import os
import sys
import glob
import chromadb
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables
load_dotenv()


def chunk_document(text: str, source_name: str) -> list[dict]:
    """
    Split a document into chunks by paragraph.
    Returns a list of dicts with text and metadata.
    """
    paragraphs = text.strip().split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        # Skip short lines (headers, separators)
        if len(para) < 50:
            continue
        # Skip separator lines
        if para.startswith("===="):
            continue

        chunks.append({
            "text": para,
            "source": source_name
        })

    return chunks


def load_documents(data_dir: str = "data") -> list[dict]:
    """
    Load all text documents from data directory and chunk them.
    """
    all_chunks = []
    
    file_source_map = {
        "company_hr_policy.txt": "HR Policy",
        "engineering_standards.txt": "Engineering",
        "onboarding_guide.txt": "Onboarding",
        "product_knowledge_base.txt": "Product",
        "security_policy.txt": "Security"
    }

    if not os.path.exists(data_dir):
        data_dir = "."  # fallback to current directory

    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt documents found in '{data_dir}' directory.")

    print(f"[*] Loading {len(txt_files)} documents from '{data_dir}'...")

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        source_name = file_source_map.get(filename, filename.replace(".txt", "").replace("_", " ").title())

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_document(content, source_name)
        all_chunks.extend(chunks)
        print(f"  + Loaded '{filename}' as [{source_name}] ({len(chunks)} chunks)")

    print(f"[*] Total chunks created: {len(all_chunks)}")
    return all_chunks


def run_ingestion(db_path: str = "./chroma_db", collection_name: str = "company_docs"):
    """
    Ingest document chunks into persistent ChromaDB collection.
    """
    chunks = load_documents()

    print(f"\n[*] Initializing Chroma persistent client at '{db_path}'...")
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Re-create collection to prevent duplicate entries on re-runs
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"  Existing collection '{collection_name}' reset.")
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    documents = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"]} for c in chunks]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print(f"[OK] Successfully ingested {len(documents)} chunks into ChromaDB collection '{collection_name}'!")


if __name__ == "__main__":
    run_ingestion()
