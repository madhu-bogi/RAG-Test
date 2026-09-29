from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float = Field(..., ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str
    service: str
