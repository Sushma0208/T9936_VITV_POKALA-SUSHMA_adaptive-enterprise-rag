from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.hybrid_search import HybridSearch
from app.verification.evidence_checker import EvidenceChecker


def main():

    documents = load_text_documents()
    chunks = chunk_documents(documents)

    hybrid = HybridSearch(chunks)
    checker = EvidenceChecker()

    # -----------------------------------
    # ROLE-AWARE SECURITY TESTS
    # -----------------------------------

    security_tests = [
        {
            "question": "How many days can employees work from home?",
            "role": "IT",
            "blocked_source": "WFH_Policy.txt"
        },
        {
            "question": "How many annual leave days are available?",
            "role": "IT",
            "blocked_source": "Leave_Policy.txt"
        },
        {
            "question": "What is the minimum enterprise password length?",
            "role": "IT",
            "allowed_source": "Password_Policy.txt"
        }
    ]

    security_correct = 0

    print("\n========== ROLE SECURITY ==========")

    for test in security_tests:

        results = hybrid.search(
            query=test["question"],
            top_k=5,
            alpha=0.5,
            user_role=test["role"]
        )

        sources = [
            result["document"]["source"]
            for result in results
        ]

        if "blocked_source" in test:

            passed = (
                test["blocked_source"]
                not in sources
            )

        else:

            passed = (
                test["allowed_source"]
                in sources
            )

        if passed:
            security_correct += 1

        print("\nQuestion:", test["question"])
        print("Role:", test["role"])
        print("Retrieved:", sources)
        print("Passed:", passed)

    # -----------------------------------
    # EVIDENCE VERIFICATION TESTS
    # -----------------------------------

    evidence_sources = hybrid.search(
        query="How many days can employees work from home?",
        top_k=5,
        alpha=0.5,
        user_role="Employee"
    )

    verification_tests = [
        {
            "answer":
                "Employees can work from home for 8 days per month.",
            "expected": True
        },
        {
            "answer":
                "Employees can work from home for 25 days per month.",
            "expected": False
        }
    ]

    verification_correct = 0

    print("\n========== EVIDENCE VERIFICATION ==========")

    for test in verification_tests:

        result = checker.check(
            answer=test["answer"],
            sources=evidence_sources
        )

        passed = (
            result["supported"]
            == test["expected"]
        )

        if passed:
            verification_correct += 1

        print("\nAnswer:", test["answer"])
        print("Expected support:", test["expected"])
        print("Result:", result)
        print("Passed:", passed)

    print("\n======================================")
    print("FINAL SECURITY / VERIFICATION RESULTS")
    print("======================================")

    print(
        f"Role Security: "
        f"{security_correct}/{len(security_tests)}"
    )

    print(
        f"Evidence Verification: "
        f"{verification_correct}/{len(verification_tests)}"
    )


if __name__ == "__main__":
    main()