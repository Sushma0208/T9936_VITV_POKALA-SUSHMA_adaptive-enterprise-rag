from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts):
        """
        Convert text into numerical embeddings.

        Args:
            texts: A string or list of strings.

        Returns:
            Embedding vectors.
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )