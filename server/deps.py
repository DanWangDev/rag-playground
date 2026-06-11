"""FastAPI dependency injection — provides shared RAG pipeline components.

Key Learning Points:
  - FastAPI's Depends() creates singleton-like dependencies
  - Module-level lazy init avoids loading models at import time
  - The vector store is kept in memory for the server lifetime
"""

from dataclasses import dataclass

from rag_playground.config.env import settings
from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore


@dataclass
class AppState:
    """Application state shared across all request handlers."""

    vector_store: InMemoryVectorStore


_state: AppState | None = None


def get_state() -> AppState:
    """Get or create the shared application state."""
    global _state
    if _state is None:
        _state = AppState(
            vector_store=InMemoryVectorStore(
                embedding_model=settings.embed_model,
                dimensions=settings.embed_dimensions,
            ),
        )
    return _state


def reset_state() -> None:
    """Reset the application state (useful for testing)."""
    global _state
    _state = None
