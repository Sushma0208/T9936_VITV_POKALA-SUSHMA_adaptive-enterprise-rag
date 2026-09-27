from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.rag.hybrid_rag_pipeline import HybridRAGPipeline


def create_pipeline():

    documents = load_text_documents()

    chunks = chunk_documents(documents)

    return HybridRAGPipeline(chunks)


def test_hybrid_rag_wfh():

    rag = create_pipeline()

    result = rag.answer(
        question="How many days can employees work from home?",
        user_role="Employee"
    )

    print("\n--- Hybrid RAG Answer ---")
    print(result["answer"])

    print("\n--- Sources ---")

    for source in result["sources"]:
        print(
            source["document"]["source"],
            "Hybrid Score:",
            round(source["score"], 4)
        )

    assert result["answer"]
    assert len(result["sources"]) > 0


def test_hybrid_rag_unsupported():

    rag = create_pipeline()

    result = rag.answer(
        question="What is the company's maternity leave policy?",
        user_role="Employee"
    )

    print("\n--- Unsupported Question ---")
    print(result["answer"])

    assert result["answer"]