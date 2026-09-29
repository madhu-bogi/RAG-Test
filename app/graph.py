from __future__ import annotations
from typing import TypedDict
import json
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from .config import get_settings
from .prompts import GROUNDING_PROMPT, SYSTEM_PROMPT
from .vector_store import VectorStore

REFUSAL = "I cannot answer this question because the information is not available in the provided Agentic AI eBook."

class RAGState(TypedDict, total=False):
    query: str
    matches: list[dict]
    context: str
    final_answer: str
    grounded: bool
    confidence_score: float
    retrieved_context_chunks: list[str]

class RAGGraph:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = VectorStore()
        self.llm = OpenAI(api_key=self.settings["openai_api_key"])
        self.graph = self._build_graph()

    def _retrieve(self, state: RAGState) -> RAGState:
        matches = self.store.query(state["query"], self.settings["top_k"])
        relevant = [m for m in matches if m["score"] >= self.settings["min_relevance_score"]]
        return {"matches": relevant, "retrieved_context_chunks": [m["text"] for m in relevant], "context": self._format_context(relevant)}

    @staticmethod
    def _format_context(matches: list[dict]) -> str:
        return "\n\n".join(f"[Chunk {i} | Page {m['page']} | Similarity {m['score']:.4f}]\n{m['text']}" for i, m in enumerate(matches, 1))

    def _route_after_retrieval(self, state: RAGState) -> str:
        return "generate" if state.get("matches") else "refuse"

    def _generate(self, state: RAGState) -> RAGState:
        response = self.llm.chat.completions.create(
            model=self.settings["chat_model"], temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"CONTEXT:\n{state['context']}\n\nQUESTION:\n{state['query']}"}],
        )
        return {"final_answer": (response.choices[0].message.content or REFUSAL).strip()}

    def _grade(self, state: RAGState) -> RAGState:
        response = self.llm.chat.completions.create(
            model=self.settings["chat_model"], temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": GROUNDING_PROMPT.format(
                question=state["query"], context=state["context"], answer=state["final_answer"])}],
        )
        try:
            result = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            result = {"grounded": False}
        grounded = bool(result.get("grounded", False))
        scores = [m["score"] for m in state["matches"]]
        avg_score = sum(scores[:3]) / min(3, len(scores)) if scores else 0.0
        confidence = 0.65 * avg_score + 0.35 * float(grounded)
        if not grounded:
            return {"grounded": False, "final_answer": REFUSAL, "confidence_score": 0.0}
        return {"grounded": True, "confidence_score": round(max(0.0, min(0.99, confidence)), 4)}

    def _refuse(self, state: RAGState) -> RAGState:
        return {"final_answer": REFUSAL, "grounded": False, "confidence_score": 0.0, "retrieved_context_chunks": []}

    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("grade", self._grade)
        builder.add_node("refuse", self._refuse)
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges("retrieve", self._route_after_retrieval, {"generate": "generate", "refuse": "refuse"})
        builder.add_edge("generate", "grade")
        builder.add_edge("grade", END)
        builder.add_edge("refuse", END)
        return builder.compile()

    def ask(self, query: str) -> dict:
        result = self.graph.invoke({"query": query.strip()})
        return {"query": query, "final_answer": result.get("final_answer", REFUSAL),
                "retrieved_context_chunks": result.get("retrieved_context_chunks", []),
                "confidence_score": float(result.get("confidence_score", 0.0))}
