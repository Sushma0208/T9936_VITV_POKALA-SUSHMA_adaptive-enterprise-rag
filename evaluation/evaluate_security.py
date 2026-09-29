from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.hybrid_search import HybridSearch


def create_hybrid_search():
    documents = load_documents()
    chunks = chunk_documents(documents)
    return HybridSearch(chunks)


def test_it_user_cannot_access_hr_wfh():
    hybrid_search = create_hybrid_search()

    results = hybrid_search.search(
        query="How many days can employees work from home?",
        top_k=5,
        alpha=0.5,
        user_role="IT"
    )

    restricted_sources = [
        result["document"]["source"]
        for result in results
        if result["document"]["access_type"] == "restricted"
    ]

    assert "WFH_Policy.txt" not in restricted_sources


def test_employee_can_access_hr_wfh():
    hybrid_search = create_hybrid_search()

    results = hybrid_search.search(
        query="How many days can employees work from home?",
        top_k=5,
        alpha=0.5,
        user_role="Employee"
    )

    retrieved_sources = [
        result["document"]["source"]
        for result in results
    ]

    assert "WFH_Policy.txt" in retrieved_sources


def test_public_documents_accessible_to_it():
    hybrid_search = create_hybrid_search()

    results = hybrid_search.search(
        query="What are the risks of using personal devices?",
        top_k=5,
        alpha=0.5,
        user_role="IT"
    )

    retrieved_sources = [
        result["document"]["source"]
        for result in results
    ]

    assert "byod-policy.pdf" in retrieved_sources


if __name__ == "__main__":
    test_it_user_cannot_access_hr_wfh()
    print("IT unauthorized access test: PASSED")

    test_employee_can_access_hr_wfh()
    print("Employee authorized access test: PASSED")

    test_public_documents_accessible_to_it()
    print("Public document access test: PASSED")

    print("\nAll security checks passed.")