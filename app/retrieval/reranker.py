class SimpleReranker:

    def rerank(self, results, top_k=3):

        # Remove duplicate chunks
        unique_results = {}

        for result in results:

            document = result["document"]

            key = (
                document["source"],
                document["chunk_id"]
            )

            if key not in unique_results:
                unique_results[key] = result

        # Sort using hybrid score
        ranked_results = sorted(
            unique_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked_results[:top_k]