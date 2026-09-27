from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.hybrid_search import HybridSearch
from evaluation.evaluation_questions import EVALUATION_DATA


def main():

    documents = load_text_documents()
    chunks = chunk_documents(documents)

    hybrid_search = HybridSearch(chunks)

    correct = 0

    for item in EVALUATION_DATA:

        query = item["question"]
        expected_source = item["expected_source"]

        results = hybrid_search.search(
            query=query,
            top_k=3,
            alpha=0.5
        )

        retrieved_sources = [
            result["document"]["source"]
            for result in results
        ]

        if expected_source in retrieved_sources:
            correct += 1

        print("\nQuestion:", query)
        print("Expected:", expected_source)
        print("Retrieved:", retrieved_sources)

    recall = correct / len(EVALUATION_DATA)

    print("\n==============================")
    print(
        f"Hybrid Recall@3: {recall * 100:.2f}%"
    )
    print(
        f"Correct: {correct}/{len(EVALUATION_DATA)}"
    )
    print("==============================")


if __name__ == "__main__":
    main()