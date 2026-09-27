from app.rag.llm import LocalLLM


def test_llm():
    llm = LocalLLM()

    response = llm.generate(
        "Answer in one sentence: What is an enterprise knowledge assistant?"
    )

    print("\nLLM Response:")
    print(response)

    assert response