import sys
import time
from dotenv import load_dotenv
from query import ask_rag, rag_agent

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


TEST_SUITE = [
    {
        "category": "HR Policy",
        "question": "What is the work from home policy?",
        "expected_keywords": ["probation", "2 days per week", "Wednesday and Friday", "10:00 AM to 6:00 PM"]
    },
    {
        "category": "HR Policy",
        "question": "How many sick leave days do full-time employees get per year?",
        "expected_keywords": ["12 days", "medical certificate"]
    },
    {
        "category": "Product Knowledge Base",
        "question": "What are the pricing plans for CloudDesk Pro?",
        "expected_keywords": ["Starter", "Business", "Enterprise", "299", "699"]
    },
    {
        "category": "Security Policy",
        "question": "What is the password policy for company accounts?",
        "expected_keywords": ["14 characters", "uppercase", "special character", "90 days"]
    },
    {
        "category": "Engineering Standards",
        "question": "What Python styling guide and code quality tools are required?",
        "expected_keywords": ["PEP 8", "Black", "Flake8", "Ruff"]
    },
    {
        "category": "Out-of-Domain (Unanswerable)",
        "question": "What is the capital of France?",
        "expected_keywords": ["don't have enough context", "not enough information"]
    }
]


def run_evaluation():
    print("=" * 70)
    print("      Running RAG Pipeline Test Suite Evaluation")
    print("=" * 70)

    total_tests = len(TEST_SUITE)
    passed_tests = 0

    for idx, test in enumerate(TEST_SUITE, 1):
        category = test["category"]
        question = test["question"]
        expected = test["expected_keywords"]

        print(f"\nTest {idx}/{total_tests} [{category}]")
        print(f"[?] Question: {question}")

        start_time = time.time()
        try:
            answer = ask_rag(question, n_results=3, verbose=False)
            elapsed = time.time() - start_time

            # Check keyword match
            matches = [kw for kw in expected if kw.lower() in answer.lower()]
            success = len(matches) > 0

            if success:
                passed_tests += 1
                print(f"[OK] PASSED ({elapsed:.2f}s) | Matched keywords: {matches}")
            else:
                print(f"[X] FAILED ({elapsed:.2f}s) | Expected any of: {expected}")
                print(f"    Received Answer: {answer[:120]}...")

        except Exception as e:
            print(f"[!] ERROR executing test: {e}")

    print("\n" + "=" * 70)
    print(f"[*] Evaluation Complete: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
