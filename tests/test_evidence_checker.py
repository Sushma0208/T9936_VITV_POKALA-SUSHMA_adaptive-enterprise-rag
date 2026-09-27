from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.hybrid_search import HybridSearch
from app.verification.evidence_checker import EvidenceChecker


def create_sources():

    documents = load_text_documents()
    chunks = chunk_documents(documents)

    hybrid = HybridSearch(chunks)

    return hybrid.search(
        query="How many days can employees work from home?",
        top_k=5,
        alpha=0.5,
        user_role="Employee"
    )


def test_supported_answer():

    sources = create_sources()

    checker = EvidenceChecker()

    result = checker.check(
        answer=(
            "Employees can work from home for "
            "8 days per month."
        ),
        sources=sources
    )

    print("\n--- Supported Answer ---")
    print(result)

    assert result["supported"] is True
    assert result["confidence"] > 0


def test_unsupported_answer():

    sources = create_sources()

    checker = EvidenceChecker()

    result = checker.check(
        answer=(
            "Employees can work from home for "
            "25 days per month."
        ),
        sources=sources
    )

    print("\n--- Unsupported Answer ---")
    print(result)

    assert result["supported"] is False