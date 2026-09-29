from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Search


documents = load_text_documents()

chunks = chunk_documents(documents)

search_engine = BM25Search(chunks)

query = "How many days can employees work remotely?"

results = search_engine.search(
    query,
    top_k=3
)

print("\nQuery:", query)

for i, result in enumerate(results, start=1):

    document = result["document"]

    print("\n==============================")
    print("Result:", i)
    print("Score:", round(result["score"], 4))
    print("Source:", document["source"])
    print("Organization:", document["organization"])
    print("Text:")
    print(document["text"])