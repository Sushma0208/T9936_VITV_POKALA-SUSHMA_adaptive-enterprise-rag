from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents


documents = load_text_documents()

chunks = chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

for chunk in chunks:
    print("\n==============================")
    print("Chunk ID:", chunk["chunk_id"])
    print("Source:", chunk["source"])
    print("Department:", chunk["department"])
    print("Allowed Roles:", chunk["allowed_roles"])
    print("Chunk Text:")
    print(chunk["text"])