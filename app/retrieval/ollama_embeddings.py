from langchain_ollama import OllamaEmbeddings

MODEL_NAME = "nomic-embed-text"


class OllamaEmbeddingModel:

    def __init__(self):
        self.model = OllamaEmbeddings(
            model=MODEL_NAME
        )

    def encode(self, texts, batch_size=20):
        """
        Generate embeddings in batches to avoid
        sending a very large corpus to Ollama at once.
        """

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                start:start + batch_size
            ]

            batch_embeddings = (
                self.model.embed_documents(batch)
            )

            embeddings.extend(
                batch_embeddings
            )

        return embeddings

    def encode_query(self, query):
        return self.model.embed_query(query)