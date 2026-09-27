from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Search

from evaluation.evaluation_questions import EVALUATION_DATA


documents = load_text_documents()

chunks = chunk_documents(documents)

search_engine = BM25Search(chunks)

correct = 0

print("\n========== EVALUATION ==========\n")

for item in EVALUATION_DATA:

    question = item["question"]
    expected = item["expected_source"]

    results = search_engine.search(
        question,
        top_k=3
    )

    retrieved_sources = [
        result["document"]["source"]
        for result in results
    ]

    success = expected in retrieved_sources

    if success:
        correct += 1

    print("Question:", question)
    print("Expected:", expected)
    print("Retrieved:", retrieved_sources)
    print("Correct:", success)
    print("--------------------------------")

recall_at_3 = correct / len(EVALUATION_DATA)

print("\n========== FINAL RESULT ==========")

print(
    f"Recall@3: {recall_at_3:.2%}"
)

print(
    f"Correct: {correct}/{len(EVALUATION_DATA)}"
)