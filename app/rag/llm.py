from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


class LocalLLM:
    def __init__(self):
        self.llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content