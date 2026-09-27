from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.hybrid_search import HybridSearch


def test_hybrid_search():

    # Load documents
    documents = load_text_documents()

    # Create chunks
    chunks = chunk_documents(documents)

    # Initialize hybrid search
    hybrid_search = HybridSearch(chunks)

    # Test query
    query = "How many days can employees work remotely?"

    results = hybrid_search.search(
        query,
        top_k=5,
        alpha=0.5
    )

    print("\n--- Hybrid Search Results ---")

    for i, result in enumerate(results, start=1):

        document = result["document"]

        print(f"\nResult {i}")
        print("Hybrid Score:", round(result["score"], 4))
        print("BM25 Score:", round(result["bm25_score"], 4))
        print("Vector Score:", round(result["vector_score"], 4))
        print("Source:", document["source"])
        print("Department:", document["department"])
        print("Text:", document["text"][:250])

    assert len(results) > 0

    # WFH policy should be among the top results
    sources = [
        result["document"]["source"]
        for result in results
    ]

    assert "WFH_Policy.txt" in sources