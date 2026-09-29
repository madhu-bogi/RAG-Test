from fastapi.testclient import TestClient

from app.graph import REFUSAL
from app.main import app
from app.schemas import ChatRequest, ChatResponse

def test_homepage_serves_chat_frontend():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "Ask the source" in response.text

def test_request_validation():
    assert ChatRequest(query="What is Agentic AI?").query

def test_response_shape():
    response = ChatResponse(query="test", final_answer="answer", retrieved_context_chunks=["context"], confidence_score=0.8)
    assert response.confidence_score == 0.8

def test_refusal_message_exists():
    assert "provided Agentic AI eBook" in REFUSAL
