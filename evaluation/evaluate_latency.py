import time

from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.rag.hybrid_rag_pipeline import HybridRAGPipeline


def main():

    documents = load_text_documents()
    chunks = chunk_documents(documents)

    rag = HybridRAGPipeline(chunks)

    questions = [
        "How many days can employees work from home?",
        "What is the minimum enterprise password length?",
        "How many annual leave days are employees entitled to?"
    ]

    latencies = []

    print("\n========== LATENCY EVALUATION ==========")

    for question in questions:

        start_time = time.perf_counter()

        result = rag.answer(
            question=question,
            user_role="Employee"
        )

        end_time = time.perf_counter()

        latency = end_time - start_time

        latencies.append(latency)

        print("\nQuestion:", question)
        print("Latency:", round(latency, 2), "seconds")
        print("Answer:", result["answer"])

    average_latency = sum(latencies) / len(latencies)

    print("\n========================================")
    print("LATENCY RESULTS")
    print("========================================")

    print(
        "Average latency:",
        round(average_latency, 2),
        "seconds"
    )

    print(
        "Minimum latency:",
        round(min(latencies), 2),
        "seconds"
    )

    print(
        "Maximum latency:",
        round(max(latencies), 2),
        "seconds"
    )


if __name__ == "__main__":
    main()