# Submission Notes

## Before pushing to GitHub

1. Copy `.env.example` to `.env`.
2. Add your OpenAI and Pinecone keys locally.
3. Run `pip install -r requirements.txt`.
4. Run `python scripts/ingest.py`.
5. Run `uvicorn app.main:app --reload`.
6. Test `/chat` with the six assignment queries.
7. Run `pytest -q`.
8. Confirm `.env` and `data/agentic-ai.pdf` are not tracked by Git.
9. Push the repository and submit the public repository URL.

## Git commands

```bash
git init
git add .
git status
git commit -m "Build LangGraph Pinecone RAG chatbot"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/agentic-ai-rag-chatbot.git
git push -u origin main
```

## Important

Do not put API keys in the repository, README, screenshots, or commit history.
