import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, embedding_dimension):
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.documents = []

    def add_documents(self, documents, embeddings):
        """
        Add document embeddings to the FAISS index.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(self, query_embedding, top_k=3):
        """
        Search for the most similar documents.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "document": self.documents[index]
                }
            )

        return results