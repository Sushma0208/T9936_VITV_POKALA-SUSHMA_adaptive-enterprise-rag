def chunk_documents(
    documents,
    chunk_size=500,
    overlap=100,
    chunk_overlap=None
):
    if chunk_overlap is not None:
        overlap = chunk_overlap
    chunks = []

    for document in documents:
        text = document["text"].strip()

        if not text:
            continue

        start = 0
        chunk_id = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end]

            chunk = {
                "text": chunk_text,
                "source": document["source"],
                "organization": document["organization"],
                "document_type": document.get("document_type", "Unknown"),
                "file_type": document["file_type"],
                "page_number": document["page_number"],
                "allowed_roles": document.get("allowed_roles", []),
                "access_type": document.get("access_type", "restricted"),
                "chunk_id": chunk_id,
            }

            chunks.append(chunk)

            if end >= len(text):
                break

            start = end - overlap
            chunk_id += 1

    return chunks