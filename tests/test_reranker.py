from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import SimpleReranker


def test_reranker():

    documents = load_text_documents()
    chunks = chunk_documents(documents)

    hybrid = HybridSearch(chunks)

    results = hybrid.search(
        query="How many days can employees work from home?",
        top_k=5,
        alpha=0.5
    )

    reranker = SimpleReranker()

    reranked = reranker.rerank(
        results,
        top_k=3
    )

    print("\n--- Reranked Results ---")

    for result in reranked:

        document = result["document"]

        print(
            document["source"],
            "Chunk:",
            document["chunk_id"],
            "Score:",
            round(result["score"], 4)
        )

    assert len(reranked) <= 3

    # Ensure duplicate chunk IDs are removed
    keys = [
        (
            result["document"]["source"],
            result["document"]["chunk_id"]
        )
        for result in reranked
    ]

    assert len(keys) == len(set(keys))