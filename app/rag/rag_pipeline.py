from app.retrieval.bm25_search import BM25Search
from app.retrieval.role_filter import filter_by_role
from app.rag.llm import LocalLLM
from app.rag.prompt import build_rag_prompt


class RAGPipeline:

    def __init__(self, documents):
        self.search_engine = BM25Search(documents)
        self.llm = LocalLLM()

    def answer(self, question, user_role, top_k=5):

        # Step 1: Retrieve documents
        retrieved = self.search_engine.search(
            question,
            top_k=top_k
        )

        # Step 2: Apply role-based access filtering
        authorized = filter_by_role(
            retrieved,
            user_role
        )

        # Step 3: If no authorized evidence exists
        if not authorized:
            return {
                "answer": "Insufficient evidence in the available documents.",
                "sources": []
            }

        # Step 4: Build evidence-grounded prompt
        prompt = build_rag_prompt(
            question,
            authorized
        )

        # Step 5: Generate answer using local LLM
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": authorized
        }