from __future__ import annotations
import time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from .config import get_settings

class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.openai = OpenAI(api_key=self.settings["openai_api_key"])
        self.pinecone = Pinecone(api_key=self.settings["pinecone_api_key"])
        self.index_name = self.settings["index_name"]
        self.dimension = 1536
        self.index = None

    def ensure_index(self) -> None:
        names = {item["name"] for item in self.pinecone.list_indexes()}
        if self.index_name not in names:
            self.pinecone.create_index(
                name=self.index_name, dimension=self.dimension, metric="cosine",
                spec=ServerlessSpec(cloud=self.settings["pinecone_cloud"], region=self.settings["pinecone_region"]),
            )
            for _ in range(60):
                status = self.pinecone.describe_index(self.index_name).status
                ready = status.get("ready", False) if isinstance(status, dict) else getattr(status, "ready", False)
                if ready:
                    break
                time.sleep(2)
        self.index = self.pinecone.Index(self.index_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.embeddings.create(model=self.settings["embedding_model"], input=texts)
        return [item.embedding for item in response.data]

    def upsert_chunks(self, chunks: list[dict]) -> int:
        self.ensure_index()
        batch_size = self.settings["embedding_batch_size"]
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            embeddings = self.embed_texts([item["text"] for item in batch])
            vectors = []
            for item, embedding in zip(batch, embeddings):
                vectors.append({"id": item["id"], "values": embedding, "metadata": {
                    "text": item["text"], "page": item["page"], "chunk_index": item["chunk_index"], "source": item["source"]
                }})
            self.index.upsert(vectors=vectors)
            total += len(vectors)
        return total

    def query(self, query: str, top_k: int | None = None) -> list[dict]:
        self.ensure_index()
        embedding = self.embed_texts([query])[0]
        result = self.index.query(vector=embedding, top_k=top_k or self.settings["top_k"], include_metadata=True)
        matches = []
        for match in result.matches:
            metadata = match.metadata or {}
            matches.append({"id": match.id, "score": float(match.score), "text": str(metadata.get("text", "")),
                            "page": int(metadata.get("page", 0)), "chunk_index": int(metadata.get("chunk_index", 0)),
                            "source": str(metadata.get("source", ""))})
        return matches
