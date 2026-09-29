from functools import lru_cache
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "agentic-ai.pdf"
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

@lru_cache
def get_settings() -> dict:
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
    return {
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "pinecone_api_key": os.environ["PINECONE_API_KEY"],
        "index_name": os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag"),
        "pinecone_cloud": os.getenv("PINECONE_CLOUD", "aws"),
        "pinecone_region": os.getenv("PINECONE_REGION", "us-east-1"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chat_model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "top_k": int(os.getenv("TOP_K", "5")),
        "min_relevance_score": float(os.getenv("MIN_RELEVANCE_SCORE", "0.35")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
    }
