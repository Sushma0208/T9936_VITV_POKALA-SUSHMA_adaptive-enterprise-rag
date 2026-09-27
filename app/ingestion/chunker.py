from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Split documents into meaningful overlapping chunks.

    The chunker tries to preserve complete paragraphs/sections
    instead of cutting words in the middle.

    Args:
        documents: Loaded documents with metadata.
        chunk_size: Maximum approximate characters per chunk.
        chunk_overlap: Approximate overlap between chunks.

    Returns:
        List of chunks with preserved metadata.
    """

    chunks = []

    for document in documents:

        # Split using blank lines so that sections remain meaningful
        paragraphs = [
            paragraph.strip()
            for paragraph in document["text"].split("\n\n")
            if paragraph.strip()
        ]

        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:

            # If adding the next paragraph stays within the limit
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

            else:
                # Store the current chunk
                if current_chunk:
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "text": current_chunk,
                            "source": document["source"],
                            "department": document["department"],
                            "allowed_roles": document["allowed_roles"],
                        }
                    )

                    chunk_id += 1

                # Start a new chunk
                current_chunk = paragraph

        # Store the final chunk
        if current_chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": current_chunk,
                    "source": document["source"],
                    "department": document["department"],
                    "allowed_roles": document["allowed_roles"],
                }
            )

    return chunks