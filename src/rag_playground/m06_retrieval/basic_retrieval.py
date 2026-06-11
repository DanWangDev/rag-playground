"""Basic retrieval: embed query → search vector store → return results.

Key Learning Points:
  - The simplest retrieval strategy — good baseline
  - Measures elapsed time for comparison with advanced strategies
  - Uses the same provider/store abstractions as all other modules
  - Optional provider params enable testing without a real Ollama server
"""

import time

from rag_playground.config.providers.base import ModelProvider
from rag_playground.m04_embeddings.embed import embed_query
from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.shared.types import RetrievalResult


async def retrieve(
    query: str,
    store: InMemoryVectorStore,
    top_k: int = 3,
    min_score: float | None = None,
    metadata_filter: dict | None = None,
    embed_provider: ModelProvider | None = None,
    embed_model: str | None = None,
) -> RetrievalResult:
    """Basic retrieval: embed the query and search the vector store.

    Args:
        query: The user's question.
        store: The vector store to search.
        top_k: Number of results to return.
        min_score: Optional minimum similarity threshold.
        metadata_filter: Optional metadata filter.
        embed_provider: Optional provider (uses config default if None).
        embed_model: Optional model name (uses config default if None).

    Returns:
        RetrievalResult with ranked search results and timing info.
    """
    start = time.perf_counter()

    # Embed the query
    query_vector = await embed_query(query, provider=embed_provider, model=embed_model)

    # Search the store
    search_results = store.search(
        query_vector,
        top_k=top_k,
        min_score=min_score,
        metadata_filter=metadata_filter,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return RetrievalResult(
        query=query,
        results=search_results,
        strategy="basic",
        elapsed_ms=elapsed_ms,
    )
