import sys
import argparse
from dotenv import load_dotenv
from ingest import run_ingestion
from query import ask_rag, rag_agent

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def print_banner():
    print("=" * 65)
    print("      Simple RAG Pipeline Assistant (Gemini + ChromaDB)")
    print("      Built without LangChain framework dependencies")
    print("=" * 65)


def interactive_cli():
    print_banner()
    print("Select Mode:")
    print("  [1] Standard Grounded RAG (Strict Document Search)")
    print("  [2] Agentic RAG (Gemini Tool Calling Agent)")
    print("  [3] Re-run Document Ingestion")
    print("  [0] Exit\n")

    choice = input("Enter choice (1/2/3/0) [Default: 1]: ").strip() or "1"

    if choice == "3":
        run_ingestion()
        return
    elif choice == "0":
        print("Goodbye!")
        sys.exit(0)

    use_agent = (choice == "2")
    mode_name = "Agentic RAG Mode" if use_agent else "Standard Grounded RAG Mode"
    print(f"\n[*] Switched to: {mode_name}")
    print("Type your questions below (or 'exit' / 'q' to quit, 'switch' to change mode):\n")

    while True:
        try:
            user_input = input("[?] Enter Question: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting application. Have a great day!")
                break

            if user_input.lower() in ["switch", "mode"]:
                use_agent = not use_agent
                mode_name = "Agentic RAG Mode" if use_agent else "Standard Grounded RAG Mode"
                print(f"[*] Switched to: {mode_name}\n")
                continue

            if use_agent:
                rag_agent(user_input, verbose=True)
            else:
                ask_rag(user_input, n_results=3, verbose=True)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"[X] Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Simple RAG Pipeline Application")
    parser.add_argument("--query", "-q", type=str, help="Single query to ask")
    parser.add_argument("--agent", "-a", action="store_true", help="Use Agentic RAG mode")
    parser.add_argument("--ingest", "-i", action="store_true", help="Run ingestion before querying")

    args = parser.parse_args()

    if args.ingest:
        run_ingestion()

    if args.query:
        print_banner()
        if args.agent:
            rag_agent(args.query, verbose=True)
        else:
            ask_rag(args.query, n_results=3, verbose=True)
    else:
        interactive_cli()


if __name__ == "__main__":
    main()
