from rank_bm25 import BM25Okapi


class BM25Search:

    def __init__(self, documents):
        """
        Initialize BM25 keyword search.

        Args:
            documents: List of document chunks.
        """

        self.documents = documents

        tokenized_documents = [
            document["text"].lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=3):

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            results.append(
                {
                    "score": float(scores[index]),
                    "document": self.documents[index]
                }
            )

        return results