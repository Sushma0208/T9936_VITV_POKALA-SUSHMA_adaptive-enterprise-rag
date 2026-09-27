from app.retrieval.hybrid_search import HybridSearch
from app.rag.llm import LocalLLM
from app.retrieval.reranker import SimpleReranker
from app.rag.prompt import build_rag_prompt
from app.verification.evidence_checker import EvidenceChecker


class HybridRAGPipeline:

    def __init__(self, documents):

        self.hybrid_search = HybridSearch(documents)
        self.llm = LocalLLM()
        self.evidence_checker = EvidenceChecker()
        self.reranker = SimpleReranker()

    def answer(
        self,
        question,
        user_role,
        top_k=5,
        alpha=0.5
    ):

        # 1. Hybrid retrieval + role filtering
        authorized_results = self.hybrid_search.search(
            query=question,
            top_k=top_k,
            alpha=alpha,
            user_role=user_role
        )

        authorized_results = self.reranker.rerank(
        authorized_results,
        top_k=top_k
        )


        # 2. No authorized evidence
        if not authorized_results:

            return {
                "answer": (
                    "Insufficient evidence in the available documents."
                ),
                "sources": [],
                "verification": {
                    "supported": False,
                    "confidence": 0.0,
                    "reason": "No authorized evidence was retrieved."
                }
            }

        # 3. Build RAG prompt
        prompt = build_rag_prompt(
            question,
            authorized_results
        )

        # 4. Generate answer
        generated_answer = self.llm.generate(prompt)

        # 5. Verify generated answer
        verification = self.evidence_checker.check(
            answer=generated_answer,
            sources=authorized_results
        )

        # 6. Handle unsupported answer
        if not verification["supported"]:

            final_answer = (
                "Insufficient evidence in the available documents."
            )

        else:

            final_answer = generated_answer

        return {
            "answer": final_answer,
            "sources": authorized_results,
            "verification": verification
        }