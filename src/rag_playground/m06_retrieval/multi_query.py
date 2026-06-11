"""Multi-query retrieval with Reciprocal Rank Fusion (RRF).

Key Learning Points:
  - Generate multiple query variants, search for each, fuse results
  - RRF gives each document a score based on its rank across all queries
  - Multi-query improves recall at the cost of more embedding calls
"""

import time

from rag_playground.m04_embeddings.embed import embed_query
from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.shared.types import RetrievalResult, SearchResult


async def multi_query_retrieve(
    queries: list[str],
    store: InMemoryVectorStore,
    top_k: int = 3,
    rrf_k: int = 60,
) -> RetrievalResult:
    """Retrieve using multiple query variants, fused with RRF.

    Reciprocal Rank Fusion:
      RRF_score(chunk) = sum over queries: 1 / (rrf_k + rank_in_query)

    Args:
        queries: Multiple query variants of the same question.
        store: The vector store to search.
        top_k: Number of results per query (before fusion).
        rrf_k: RRF constant (60 is standard).

    Returns:
        RetrievalResult with fused, deduplicated results.
    """
    start = time.perf_counter()

    all_results: dict[str, list[tuple[int, SearchResult]]] = {}

    for q in queries:
        q_vec = await embed_query(q)
        results = store.search(q_vec, top_k=top_k)
        for r in results:
            chunk_id = r.chunk.id
            if chunk_id not in all_results:
                all_results[chunk_id] = []
            all_results[chunk_id].append((r.rank, r))

    # Compute RRF scores
    fused: list[SearchResult] = []
    for chunk_id, occurrences in all_results.items():
        rrf_score = sum(1.0 / (rrf_k + rank) for rank, _ in occurrences)
        # Use the best individual score as the result score
        best = max(occurrences, key=lambda x: x[1].score)
        fused.append(
            SearchResult(
                chunk=best[1].chunk,
                score=rrf_score,  # RRF score (not cosine similarity)
                rank=0,
            )
        )

    # Sort by RRF score descending
    fused.sort(key=lambda r: r.score, reverse=True)

    # Assign ranks
    for rank, result in enumerate(fused, 1):
        result.rank = rank

    elapsed_ms = (time.perf_counter() - start) * 1000

    return RetrievalResult(
        query=" | ".join(queries),
        results=fused,
        strategy="multi_query_rrf",
        elapsed_ms=elapsed_ms,
    )
