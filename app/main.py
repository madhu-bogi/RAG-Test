from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from .graph import RAGGraph
from .ingestion import prepare_chunks
from .schemas import ChatRequest, ChatResponse, HealthResponse
from .vector_store import VectorStore

app = FastAPI(title="Agentic AI RAG Chatbot", version="1.0.0", description="Strictly grounded RAG chatbot over the Agentic AI eBook.")
FRONTEND_PATH = Path(__file__).resolve().parent / "static" / "index.html"
_rag: RAGGraph | None = None

def get_rag() -> RAGGraph:
    global _rag
    if _rag is None:
        _rag = RAGGraph()
    return _rag

@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_PATH, media_type="text/html")

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="agentic-ai-rag")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(**get_rag().ask(request.query))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/ingest")
def ingest() -> dict:
    try:
        chunks = prepare_chunks()
        count = VectorStore().upsert_chunks(chunks)
        return {"status": "success", "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
