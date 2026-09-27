from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.rag.rag_pipeline import RAGPipeline


def create_rag():

    documents = load_text_documents()
    chunks = chunk_documents(documents)

    return RAGPipeline(chunks)


def test_supported_wfh_question():

    rag = create_rag()

    result = rag.answer(
        question="How many days can employees work from home?",
        user_role="Employee"
    )

    print("\n--- Supported WFH Question ---")
    print("Answer:", result["answer"])

    assert result["answer"]
    assert len(result["sources"]) > 0


def test_supported_password_question():

    rag = create_rag()

    result = rag.answer(
        question="What is the minimum enterprise password length?",
        user_role="Employee"
    )

    print("\n--- Supported Password Question ---")
    print("Answer:", result["answer"])

    assert result["answer"]
    assert len(result["sources"]) > 0


def test_unsupported_question():

    rag = create_rag()

    result = rag.answer(
        question="What is the company's maternity leave policy?",
        user_role="Employee"
    )

    print("\n--- Unsupported Question ---")
    print("Answer:", result["answer"])

    assert result["answer"]