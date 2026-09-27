from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Search
from app.retrieval.ollama_embeddings import OllamaEmbeddingModel
from app.retrieval.vector_search import FAISSVectorStore
from app.retrieval.hybrid_search import HybridSearch
from evaluation.paraphrase_questions import PARAPHRASE_DATA


def evaluate_bm25(documents):
    search = BM25Search(documents)

    correct = 0

    for item in PARAPHRASE_DATA:

        results = search.search(
            item["question"],
            top_k=3
        )

        sources = [
            result["document"]["source"]
            for result in results
        ]

        if item["expected_source"] in sources:
            correct += 1

        print(
            f"\nBM25 | {item['question']}"
        )
        print("Expected:", item["expected_source"])
        print("Retrieved:", sources)

    return correct


def evaluate_vector(documents):

    embedding_model = OllamaEmbeddingModel()

    embeddings = embedding_model.encode(
        [document["text"] for document in documents]
    )

    vector_store = FAISSVectorStore(
        embedding_dimension=len(embeddings[0])
    )

    vector_store.add_documents(
        documents,
        embeddings
    )

    correct = 0

    for item in PARAPHRASE_DATA:

        query_embedding = embedding_model.encode_query(
            item["question"]
        )

        results = vector_store.search(
            query_embedding,
            top_k=3
        )

        sources = [
            result["document"]["source"]
            for result in results
        ]

        if item["expected_source"] in sources:
            correct += 1

        print(
            f"\nVector | {item['question']}"
        )
        print("Expected:", item["expected_source"])
        print("Retrieved:", sources)

    return correct


def evaluate_hybrid(documents):

    hybrid = HybridSearch(documents)

    correct = 0

    for item in PARAPHRASE_DATA:

        results = hybrid.search(
            query=item["question"],
            top_k=3,
            alpha=0.5
        )

        sources = [
            result["document"]["source"]
            for result in results
        ]

        if item["expected_source"] in sources:
            correct += 1

        print(
            f"\nHybrid | {item['question']}"
        )
        print("Expected:", item["expected_source"])
        print("Retrieved:", sources)

    return correct


def main():

    documents = load_text_documents()

    chunks = chunk_documents(documents)

    total = len(PARAPHRASE_DATA)

    print("\n========================================")
    print("PARAPHRASED RETRIEVAL EVALUATION")
    print("========================================")

    print("\n\n========== BM25 ==========")

    bm25_correct = evaluate_bm25(chunks)

    print("\n\n========== VECTOR ==========")

    vector_correct = evaluate_vector(chunks)

    print("\n\n========== HYBRID ==========")

    hybrid_correct = evaluate_hybrid(chunks)

    print("\n\n========================================")
    print("FINAL RESULTS")
    print("========================================")

    print(
        f"BM25 Recall@3: "
        f"{(bm25_correct / total) * 100:.2f}% "
        f"({bm25_correct}/{total})"
    )

    print(
        f"Vector Recall@3: "
        f"{(vector_correct / total) * 100:.2f}% "
        f"({vector_correct}/{total})"
    )

    print(
        f"Hybrid Recall@3: "
        f"{(hybrid_correct / total) * 100:.2f}% "
        f"({hybrid_correct}/{total})"
    )


if __name__ == "__main__":
    main()