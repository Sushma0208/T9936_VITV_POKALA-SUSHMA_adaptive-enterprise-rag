from app.retrieval.ollama_embeddings import OllamaEmbeddingModel


def test_ollama_embeddings():

    embedding_model = OllamaEmbeddingModel()

    embeddings = embedding_model.encode(
        [
            "Employees can work from home for 8 days.",
            "Enterprise passwords must contain at least 12 characters."
        ]
    )

    print("\nNumber of embeddings:", len(embeddings))
    print("Embedding dimension:", len(embeddings[0]))

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0