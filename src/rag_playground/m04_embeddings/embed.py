"""Embedding generation via the model provider.

Key Learning Points:
  - embed_query() generates a single vector for a search query
  - embed_documents() batch-generates vectors for multiple texts
  - The provider abstraction means this code works with any backend
  - Batch embedding is significantly faster than N individual calls
  - Optional provider parameter enables testing without Ollama
"""

from rag_playground.config.models import resolve_embed_config
from rag_playground.config.providers.base import ModelProvider


async def embed_query(
    text: str, provider: ModelProvider | None = None, model: str | None = None
) -> list[float]:
    """Generate an embedding vector for a single query.

    Args:
        text: The query text (or any single piece of text).
        provider: Optional ModelProvider (uses default from config if None).
        model: Optional model name (uses default from config if None).

    Returns:
        A list of floats representing the embedding vector.
    """
    if provider is None:
        provider, model_resolved, _ = resolve_embed_config()
        model = model or model_resolved
    elif model is None:
        _, model, _ = resolve_embed_config()
    return await provider.embed_single(model, text)


async def embed_documents(
    texts: list[str],
    provider: ModelProvider | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Generate embedding vectors for multiple documents.

    Uses batch embedding for efficiency — one API call instead of N.

    Args:
        texts: List of document texts to embed.
        provider: Optional ModelProvider (uses default from config if None).
        model: Optional model name (uses default from config if None).

    Returns:
        List of embedding vectors (one per input text).
    """
    if not texts:
        return []

    if provider is None:
        provider, model_resolved, _ = resolve_embed_config()
        model = model or model_resolved
    elif model is None:
        _, model, _ = resolve_embed_config()

    result = await provider.embed(model, texts)
    return result.vectors
