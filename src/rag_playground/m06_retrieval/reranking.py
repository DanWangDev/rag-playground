"""LLM-based reranking of retrieval results.

Key Learning Points:
  - Retrieve more candidates than needed (e.g., 20), then rerank down to top-k
  - LLM scores relevance of each candidate independently
  - Reranking improves precision at the cost of N additional LLM calls
"""

from rag_playground.config.providers.base import ChatMessage
from rag_playground.shared.types import SearchResult


async def rerank_with_llm(
    query: str,
    candidates: list[SearchResult],
    chat_fn,
    top_k: int = 3,
) -> list[SearchResult]:
    """Rerank candidates using an LLM relevance scorer.

    For each candidate, asks the LLM: "On a scale of 1-10, how relevant
    is this passage to the query?" Scores are normalized and used to reorder.

    Args:
        query: The user's question.
        candidates: List of retrieved chunks to rerank.
        chat_fn: Async chat function.
        top_k: Number of final results to return.

    Returns:
        Re-ranked list of top_k SearchResults.
    """
    if not candidates:
        return []

    scored: list[SearchResult] = []

    for candidate in candidates:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a relevance judge. Rate how relevant the passage is "
                    "to the query on a scale of 1-10, where 10 is perfectly relevant "
                    "and 1 is completely irrelevant. Respond with ONLY the number."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Query: {query}\n\nPassage: {candidate.chunk.content[:500]}",
            ),
        ]
        response = await chat_fn(messages)

        # Parse score from response
        try:
            score = float(response.strip()) / 10.0  # Normalize to 0-1
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = candidate.score  # Fall back to original score

        scored.append(
            SearchResult(
                chunk=candidate.chunk,
                score=score,
                rank=0,
            )
        )

    # Re-sort by LLM score
    scored.sort(key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(scored[:top_k], 1):
        result.rank = rank

    return scored[:top_k]
