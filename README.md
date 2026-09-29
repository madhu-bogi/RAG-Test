# Agentic AI RAG Chatbot — LangGraph + Pinecone

Custom Python implementation for the Appening AI Engineer interview assignment.

The chatbot answers questions **only from the supplied Agentic AI eBook** and refuses questions when relevant evidence is not retrieved.

## Architecture

```text
Agentic AI eBook PDF
        |
        v
   PyPDF extraction
        |
        v
Recursive text splitting
(800 chars / 100 overlap)
        |
        v
OpenAI embeddings
        |
        v
Pinecone vector index
        ^
        |
User query -> FastAPI -> LangGraph
                         |
                         v
                    Retrieve Node
                         |
                  relevance threshold
                    /          \\
               relevant       none
                  |             |
                  v             v
              Generate       Refuse
                  |
                  v
             Grounding Grade
                  |
                  v
             Confidence
                  |
                  v
             JSON response
```

## Stack

- Python
- PyPDF
- LangChain RecursiveCharacterTextSplitter
- OpenAI embeddings
- Pinecone
- LangGraph
- FastAPI / Pydantic
- pytest

## Setup

### 1. Clone

```bash
git clone https://github.com/<YOUR_USERNAME>/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot
```

### 2. Virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Configure `.env`

Copy `.env.example` to `.env` and set:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Never commit `.env` or API keys.

## Ingestion

Run:

```bash
python scripts/ingest.py
```

This downloads the eBook at runtime, extracts pages, chunks text using 800 characters with 100-character overlap, embeds each chunk, and upserts vectors into Pinecone with source/page/chunk metadata.

The PDF is ignored by Git and is not redistributed in the public repository.

## Run API

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Chat frontend:

```text
http://127.0.0.1:8000/
```

## Endpoints

### `GET /health`

```json
{"status":"ok","service":"agentic-ai-rag"}
```

### `POST /ingest`

Indexes the knowledge base.

### `POST /chat`

Request:

```json
{"query":"What is Agentic AI?"}
```

Response:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Answer grounded in the retrieved eBook context...",
  "retrieved_context_chunks": ["Relevant chunk..."],
  "confidence_score": 0.87
}
```

FastAPI response models validate and document the JSON response schema.

## LangGraph workflow

```text
START
  |
  v
retrieve
  |
  +---- no relevant evidence ----> refuse ----> END
  |
  v
generate
  |
  v
grade
  |
  v
END
```

### Retrieval

The query is embedded and searched against Pinecone. Only matches above `MIN_RELEVANCE_SCORE` are passed forward.

### Generation

The LLM receives the retrieved chunks as its factual context and is instructed not to use outside knowledge.

### Grounding grade

A second LLM call evaluates whether the proposed answer is supported by the retrieved evidence. If not grounded, the final answer is replaced by the refusal response.

## Confidence score

The score is a **retrieval/grounding heuristic, not a calibrated probability**:

```text
confidence = 0.65 × average top-3 similarity
           + 0.35 × grounding_result
```

It is capped below 1.0 and is `0.0` for refusal or failed grounding.

## Out-of-scope test

For:

```text
What is the capital of France?
```

expected behavior:

```json
{
  "query": "What is the capital of France?",
  "final_answer": "I cannot answer this question because the information is not available in the provided Agentic AI eBook.",
  "retrieved_context_chunks": [],
  "confidence_score": 0.0
}
```

The system does not answer this from general model knowledge.

## Validation queries

1. What is the core definition of Agentic AI as outlined in the eBook?
2. What are the main architectural components required to build agentic systems?
3. What real-world industry use cases for Agentic AI are discussed in the eBook?
4. How does Agentic AI differ from traditional generative AI chatbots according to the text?
5. What key challenges or limitations of Agentic AI are mentioned in the document?
6. What is the capital of France?

## Tests

```bash
pytest -q
```

The included tests validate request/response models and the refusal contract. Full integration testing requires valid API credentials and a populated Pinecone index.

## Design decisions

- **FastAPI instead of Streamlit:** the assignment asks for a structured JSON payload, so an API demonstrates the backend interface directly.
- **Page metadata:** page numbers are stored with each chunk for traceability.
- **Separate grounding node:** generation and grounding are separate graph stages, making hallucination handling explicit.
- **Runtime PDF download:** the public repository does not redistribute the source eBook.
- **No hard-coded secrets:** credentials are loaded through environment variables.

## Production improvements

- Cross-encoder/reranker after vector retrieval.
- Calibrated confidence using a labeled evaluation set.
- Explicit page citations in answers.
- Hybrid BM25 + semantic retrieval.
- Automated RAG evaluation.
- Authentication/rate limiting.
- Observability for latency, tokens and retrieval quality.
- Document versioning and scheduled re-indexing.

## 60-second interview explanation

> I built a document-grounded RAG chatbot using Python, LangGraph, OpenAI embeddings, Pinecone and FastAPI. During ingestion, I download the Agentic AI eBook, extract it page-by-page, split it into overlapping chunks, generate embeddings and store those vectors with page metadata in Pinecone. At query time, LangGraph first retrieves relevant chunks. If there is no relevant evidence, the graph refuses to answer. Otherwise, the generation node answers using only the retrieved context, and a separate grounding node verifies that the answer is supported. The API returns the query, final answer, retrieved context chunks and a retrieval/grounding confidence score. This prevents the model from answering out-of-scope questions such as the capital of France from general knowledge.

## Submission checklist

- [x] Repository structure
- [x] README setup guide
- [x] PDF ingestion
- [x] Chunking with overlap
- [x] OpenAI embeddings
- [x] Pinecone vector storage
- [x] LangGraph workflow
- [x] Retrieval node
- [x] Generation node
- [x] Grounding verification
- [x] Confidence score
- [x] FastAPI API
- [x] Structured JSON response
- [x] Tests
