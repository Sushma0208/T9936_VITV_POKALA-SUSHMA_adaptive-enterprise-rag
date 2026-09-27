from app.retrieval.bm25_search import BM25Search
from app.retrieval.ollama_embeddings import OllamaEmbeddingModel
from app.retrieval.vector_search import FAISSVectorStore
from app.retrieval.role_filter import filter_by_role


class HybridSearch:

    def __init__(self, documents):

        self.documents = documents

        # BM25 keyword search
        self.bm25 = BM25Search(documents)

        # Ollama embedding model
        self.embedding_model = OllamaEmbeddingModel()

        # Generate embeddings for all documents
        embeddings = self.embedding_model.encode(
            [document["text"] for document in documents]
        )

        # FAISS vector store
        self.vector_store = FAISSVectorStore(
            embedding_dimension=len(embeddings[0])
        )

        self.vector_store.add_documents(
            documents,
            embeddings
        )

    def search(
        self,
        query,
        top_k=5,
        alpha=0.5,
        user_role=None
    ):
        """
        Perform hybrid retrieval using BM25 and vector search.

        alpha:
            Weight given to BM25.
            1 - alpha is the vector-search weight.

        user_role:
            If provided, only documents authorized for
            that role are returned.
        """

        # -----------------------------
        # 1. BM25 retrieval
        # -----------------------------
        bm25_results = self.bm25.search(
            query,
            top_k=top_k
        )

        # -----------------------------
        # 2. Vector retrieval
        # -----------------------------
        query_embedding = self.embedding_model.encode_query(
            query
        )

        vector_results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # -----------------------------
        # 3. Normalize scores
        # -----------------------------
        def normalize(scores):

            if not scores:
                return []

            minimum = min(scores)
            maximum = max(scores)

            if maximum == minimum:
                return [1.0 for _ in scores]

            return [
                (score - minimum) / (maximum - minimum)
                for score in scores
            ]

        bm25_scores = [
            result["score"]
            for result in bm25_results
        ]

        vector_scores = [
            result["score"]
            for result in vector_results
        ]

        normalized_bm25 = normalize(bm25_scores)
        normalized_vector = normalize(vector_scores)

        # -----------------------------
        # 4. Combine results
        # -----------------------------
        combined = {}

        for result, score in zip(
            bm25_results,
            normalized_bm25
        ):

            document = result["document"]

            key = (
                document["source"],
                document["chunk_id"]
            )

            combined[key] = {
                "document": document,
                "bm25_score": score,
                "vector_score": 0.0
            }

        for result, score in zip(
            vector_results,
            normalized_vector
        ):

            document = result["document"]

            key = (
                document["source"],
                document["chunk_id"]
            )

            if key not in combined:

                combined[key] = {
                    "document": document,
                    "bm25_score": 0.0,
                    "vector_score": score
                }

            else:

                combined[key]["vector_score"] = score

        # -----------------------------
        # 5. Calculate hybrid score
        # -----------------------------
        results = []

        for item in combined.values():

            hybrid_score = (
                alpha * item["bm25_score"]
                +
                (1 - alpha) * item["vector_score"]
            )

            results.append(
                {
                    "score": hybrid_score,
                    "bm25_score": item["bm25_score"],
                    "vector_score": item["vector_score"],
                    "document": item["document"]
                }
            )

        # -----------------------------
        # 6. Sort by hybrid score
        # -----------------------------
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # -----------------------------
        # 7. Role-based filtering
        # -----------------------------
        if user_role is not None:

            results = filter_by_role(
                results,
                user_role
            )

        # -----------------------------
        # 8. Return top results
        # -----------------------------
        return results[:top_k]