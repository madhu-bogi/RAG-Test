from app.ingestion import prepare_chunks
from app.vector_store import VectorStore

if __name__ == "__main__":
    chunks = prepare_chunks()
    count = VectorStore().upsert_chunks(chunks)
    print(f"Prepared {len(chunks)} chunks.")
    print(f"Indexed {count} vectors into Pinecone.")
