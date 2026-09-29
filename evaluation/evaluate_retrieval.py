import json

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Search
from app.retrieval.ollama_embeddings import OllamaEmbeddingModel
from app.retrieval.vector_search import FAISSVectorStore
from app.retrieval.hybrid_search import HybridSearch


QUESTIONS_FILE = "evaluation/questions.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_recall(results, expected_source):
    retrieved_sources = [
        result["document"]["source"]
        for result in results
    ]

    return expected_source in retrieved_sources


def main():

    print("Loading documents...")

    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    questions = load_questions()

    # -----------------------------
    # BM25
    # -----------------------------
    print("\nBuilding BM25 index...")

    bm25 = BM25Search(chunks)

    # -----------------------------
    # Vector Search
    # -----------------------------
    print("Building vector index...")

    embedding_model = OllamaEmbeddingModel()

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks]
    )

    vector_store = FAISSVectorStore(
        embedding_dimension=len(embeddings[0])
    )

    vector_store.add_documents(
        chunks,
        embeddings
    )

    # -----------------------------
    # Hybrid Search
    # -----------------------------
    print("Building Hybrid Search...")

    hybrid = HybridSearch(chunks)

    bm25_hits = 0
    vector_hits = 0
    hybrid_hits = 0

    print("\n========== RETRIEVAL EVALUATION ==========")

    for question in questions:

        query = question["question"]
        expected_source = question["expected_source"]

        bm25_results = bm25.search(
            query,
            top_k=3
        )

        query_embedding = embedding_model.encode_query(query)

        vector_results = vector_store.search(
            query_embedding,
            top_k=3
        )

        hybrid_results = hybrid.search(
            query,
            top_k=3,
            alpha=0.5
        )

        bm25_correct = evaluate_recall(
            bm25_results,
            expected_source
        )

        vector_correct = evaluate_recall(
            vector_results,
            expected_source
        )

        hybrid_correct = evaluate_recall(
            hybrid_results,
            expected_source
        )

        bm25_hits += bm25_correct
        vector_hits += vector_correct
        hybrid_hits += hybrid_correct

        print(f"\n{question['id']}: {query}")
        print(f"Expected: {expected_source}")

        print(
            "BM25:",
            "✓" if bm25_correct else "✗"
        )

        print(
            "Vector:",
            "✓" if vector_correct else "✗"
        )

        print(
            "Hybrid:",
            "✓" if hybrid_correct else "✗"
        )

    total = len(questions)

    print("\n========== RESULTS ==========")

    print(
        f"BM25 Recall@3: "
        f"{bm25_hits}/{total} "
        f"({bm25_hits / total * 100:.1f}%)"
    )

    print(
        f"Vector Recall@3: "
        f"{vector_hits}/{total} "
        f"({vector_hits / total * 100:.1f}%)"
    )

    print(
        f"Hybrid Recall@3: "
        f"{hybrid_hits}/{total} "
        f"({hybrid_hits / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()