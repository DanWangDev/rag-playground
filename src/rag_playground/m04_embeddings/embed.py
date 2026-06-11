"""Embedding generation via the model provider.

Key Learning Points:
  - embed_query() generates a single vector for a search query
  - embed_documents() batch-generates vectors for multiple texts
  - The provider abstraction means this code works with any backend
  - Batch embedding is significantly faster than N individual calls
"""

from rag_playground.config.models import resolve_embed_config


async def embed_query(text: str) -> list[float]:
    """Generate an embedding vector for a single query.

    Args:
        text: The query text (or any single piece of text).

    Returns:
        A list of floats representing the embedding vector.
    """
    provider, model, _ = resolve_embed_config()
    return await provider.embed_single(model, text)


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for multiple documents.

    Uses batch embedding for efficiency — one API call instead of N.

    Args:
        texts: List of document texts to embed.

    Returns:
        List of embedding vectors (one per input text).
    """
    if not texts:
        return []

    provider, model, _ = resolve_embed_config()
    result = await provider.embed(model, texts)
    return result.vectors
