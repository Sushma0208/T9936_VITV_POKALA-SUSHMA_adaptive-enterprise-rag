from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_search import FAISSVectorStore


# 1. Load documents
documents = load_text_documents()

# 2. Create chunks
chunks = chunk_documents(documents)

# 3. Create embedding model
embedding_model = EmbeddingModel()

# 4. Generate embeddings for chunks
chunk_texts = [chunk["text"] for chunk in chunks]
chunk_embeddings = embedding_model.encode(chunk_texts)

print("Embedding shape:", chunk_embeddings.shape)

# 5. Create FAISS index
dimension = chunk_embeddings.shape[1]

vector_store = FAISSVectorStore(dimension)

# 6. Add chunks to FAISS
vector_store.add_documents(
    chunks,
    chunk_embeddings
)

# 7. Search
query = "Can I work remotely?"

query_embedding = embedding_model.encode([query])

results = vector_store.search(
    query_embedding,
    top_k=3
)

# 8. Display results
print("\nQuery:", query)

for i, result in enumerate(results, start=1):

    document = result["document"]

    print("\n==============================")
    print("Result:", i)
    print("Score:", round(result["score"], 4))
    print("Source:", document["source"])
    print("Department:", document["department"])
    print("Text:")
    print(document["text"])