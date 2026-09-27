from langchain_ollama import OllamaEmbeddings


MODEL_NAME = "nomic-embed-text"


class OllamaEmbeddingModel:

    def __init__(self):
        self.model = OllamaEmbeddings(
            model=MODEL_NAME
        )

    def encode(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        return self.model.embed_documents(texts)

    def encode_query(self, query):

        return self.model.embed_query(query)