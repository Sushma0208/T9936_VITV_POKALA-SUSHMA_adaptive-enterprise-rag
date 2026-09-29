from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.hybrid_search import HybridSearch


def create_hybrid_search():

    documents = load_text_documents()
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

    print("\n--- IT User: WFH Query ---")

    for result in results:
        print(
            result["document"]["source"],
            "->",
            result["document"]["organization"]
        )

    # IT user must not receive the HR WFH document
    sources = [
        result["document"]["source"]
        for result in results
    ]

    assert "WFH_Policy.txt" not in sources


def test_employee_can_access_hr_wfh():

    hybrid_search = create_hybrid_search()

    results = hybrid_search.search(
        query="How many days can employees work from home?",
        top_k=5,
        alpha=0.5,
        user_role="Employee"
    )

    print("\n--- Employee: WFH Query ---")

    for result in results:
        print(
            result["document"]["source"],
            "->",
            result["document"]["organization"]
        )

    sources = [
        result["document"]["source"]
        for result in results
    ]

    assert "WFH_Policy.txt" in sources