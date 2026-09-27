from app.ingestion.loader import load_text_documents


documents = load_text_documents()

print(f"Documents loaded: {len(documents)}")

for document in documents:
    print("\n-----------------------------")
    print("Source:", document["source"])
    print("Department:", document["department"])
    print("Allowed Roles:", document["allowed_roles"])
    print("Text Preview:")
    print(document["text"][:200])