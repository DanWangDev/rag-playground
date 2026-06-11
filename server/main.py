"""FastAPI server — REST API for the RAG pipeline.

Usage:
    fastapi dev server/main.py
    python -m uvicorn server.main:app --reload

Key Learning Points:
  - FastAPI auto-generates OpenAPI docs at /docs
  - SSE (Server-Sent Events) enables streaming responses
  - CORS middleware allows the React frontend to call the API
  - Dependency injection shares the vector store across routes
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import chat, compare, config, documents

app = FastAPI(
    title="RAG Playground API",
    description="Interactive RAG pipeline — retrieve, augment, generate",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(compare.router)
app.include_router(config.router)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    from rag_playground.config.models import resolve_chat_config

    provider, model = resolve_chat_config()
    ollama_ok = await provider.health_check()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": ollama_ok,
        "model": model,
    }
