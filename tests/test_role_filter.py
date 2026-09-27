from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Search
from app.retrieval.role_filter import filter_by_role


# Load documents
documents = load_text_documents()

# Create chunks
chunks = chunk_documents(documents)

# Initialize BM25
search_engine = BM25Search(chunks)


# --------------------------------------------------
# Test Query
# --------------------------------------------------

query = "What is the minimum enterprise password length?"

results = search_engine.search(
    query,
    top_k=5
)


print("\n==============================")
print("BEFORE ROLE FILTERING")
print("==============================")

for result in results:

    document = result["document"]

    print(
        f"Source: {document['source']}"
    )

    print(
        f"Allowed Roles: {document['allowed_roles']}"
    )

    print(
        f"Score: {result['score']:.4f}"
    )

    print("------------------------------")


# --------------------------------------------------
# Apply Role Filtering
# --------------------------------------------------

user_role = "Employee"

filtered_results = filter_by_role(
    results,
    user_role
)


print("\n==============================")
print("AFTER ROLE FILTERING")
print("==============================")

print("User Role:", user_role)

for result in filtered_results:

    document = result["document"]

    print(
        f"Source: {document['source']}"
    )

    print(
        f"Allowed Roles: {document['allowed_roles']}"
    )

    print(
        f"Score: {result['score']:.4f}"
    )

    print("------------------------------")