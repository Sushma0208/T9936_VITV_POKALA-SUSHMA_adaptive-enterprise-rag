from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.ollama_embeddings import OllamaEmbeddingModel
from app.retrieval.vector_search import FAISSVectorStore


def test_vector_search_with_ollama():

    # Load documents
    documents = load_text_documents()

    # Create chunks
    chunks = chunk_documents(documents)

    # Create embedding model
    embedding_model = OllamaEmbeddingModel()

    # Generate embeddings for chunks
    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks]
    )

    # Create FAISS store
    vector_store = FAISSVectorStore(
        embedding_dimension=len(embeddings[0])
    )

    # Add chunks and embeddings
    vector_store.add_documents(
        chunks,
        embeddings
    )

    # Create query embedding
    query = "How many days can employees work remotely?"

    query_embedding = embedding_model.encode_query(query)

    # Search
    results = vector_store.search(
        query_embedding,
        top_k=3
    )

    print("\n--- Vector Search Results ---")

    for i, result in enumerate(results, start=1):

        document = result["document"]

        print(f"\nResult {i}")
        print("Score:", round(result["score"], 4))
        print("Source:", document["source"])
        print("Department:", document["department"])
        print("Text:", document["text"][:300])

    assert len(results) > 0