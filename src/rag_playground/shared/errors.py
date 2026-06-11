"""Custom exception hierarchy for RAG Playground.

Structured errors make debugging clearer: each layer of the RAG
pipeline has its own exception type, all rooted in RAGPlaygroundError.

Key Learning Points:
  - Exception hierarchies let callers catch at the right level of specificity
  - Each error carries a user-friendly message + optional underlying cause
  - Custom exceptions help distinguish "expected" failures from bugs
"""


class RAGPlaygroundError(Exception):
    """Base exception for all RAG Playground errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


# --- Provider errors ---


class ProviderError(RAGPlaygroundError):
    """Base for provider-related errors (connection, model, etc.)."""


class ConnectionError(ProviderError):
    """Cannot reach the model provider (e.g., Ollama not running)."""


class ModelNotFoundError(ProviderError):
    """Requested model is not available. Run `python scripts/pull_models.py`."""


class GenerationError(ProviderError):
    """LLM generation failed (timeout, content filter, etc.)."""


class EmbeddingError(ProviderError):
    """Embedding generation failed."""


# --- Data errors ---


class DataError(RAGPlaygroundError):
    """Base for data loading / processing errors."""


class DocumentLoadError(DataError):
    """Failed to load or parse a document file."""


class ChunkingError(DataError):
    """Failed to split a document into chunks."""


# --- Retrieval errors ---


class RetrievalError(RAGPlaygroundError):
    """Base for retrieval-related errors."""


class EmptyStoreError(RetrievalError):
    """Attempted to search an empty vector store."""


class DimensionMismatchError(RetrievalError):
    """Query embedding dimensions don't match the store's dimensions."""


# --- Validation errors ---


class ValidationError(RAGPlaygroundError):
    """Input validation failed (schema mismatch, bad parameters)."""
