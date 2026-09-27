from fastapi import FastAPI
from pydantic import BaseModel

from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.rag.hybrid_rag_pipeline import HybridRAGPipeline


app = FastAPI(
    title="Adaptive Enterprise Knowledge Assistant",
    description="Hybrid RAG based enterprise knowledge assistant",
    version="1.0.0"
)


# Load documents and initialize the RAG pipeline
documents = load_text_documents()
chunks = chunk_documents(documents)
rag_pipeline = HybridRAGPipeline(chunks)


class QueryRequest(BaseModel):
    question: str
    user_role: str = "Employee"


@app.get("/")
def root():
    return {
        "message": "Adaptive Enterprise Knowledge Assistant API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/query")
def query(request: QueryRequest):
    result = rag_pipeline.answer(
        question=request.question,
        user_role=request.user_role
    )

    return result