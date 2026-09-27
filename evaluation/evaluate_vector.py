from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.ollama_embeddings import OllamaEmbeddingModel
from app.retrieval.vector_search import FAISSVectorStore
from evaluation.evaluation_questions import EVALUATION_DATA


def main():

    documents = load_text_documents()

    chunks = chunk_documents(documents)

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

    correct = 0

    for item in EVALUATION_DATA:

        query = item["question"]
        expected_source = item["expected_source"]

        query_embedding = embedding_model.encode_query(
            query
        )

        results = vector_store.search(
            query_embedding,
            top_k=3
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
        f"Vector Recall@3: {recall * 100:.2f}%"
    )
    print(
        f"Correct: {correct}/{len(EVALUATION_DATA)}"
    )
    print("==============================")


if __name__ == "__main__":
    main()