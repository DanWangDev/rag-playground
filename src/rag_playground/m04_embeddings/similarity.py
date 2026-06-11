"""Cosine similarity and nearest-neighbor search.

Key Learning Points:
  - Cosine similarity measures the angle between two vectors
  - 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
  - find_most_similar() is a brute-force k-NN — O(n*d) for n vectors of d dimensions
  - For production, approximate nearest neighbor (ANN) is much faster
"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors.

    cosine(a, b) = dot(a, b) / (|a| * |b|)

    For normalized vectors (unit length), this simplifies to just dot(a, b).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Similarity score between -1.0 and 1.0.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions don't match: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def find_most_similar(
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Find the top-k most similar vectors to a query vector.

    Brute-force linear scan — O(n*d) where n = number of candidates,
    d = vector dimensions.

    Args:
        query_vector: The query embedding.
        candidate_vectors: List of candidate embeddings to search.
        top_k: Number of top results to return.

    Returns:
        List of (index, similarity_score) tuples, sorted by score descending.
    """
    if not candidate_vectors:
        return []

    # Compute similarity for all candidates
    scored: list[tuple[int, float]] = []
    for i, vec in enumerate(candidate_vectors):
        sim = cosine_similarity(query_vector, vec)
        scored.append((i, sim))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Compute pairwise cosine similarity for a set of vectors.

    Returns an n×n matrix where matrix[i][j] = cosine_sim(vectors[i], vectors[j]).

    Args:
        vectors: List of embedding vectors.

    Returns:
        2D list of similarity scores (n×n, symmetric).
    """
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = sim
            matrix[j][i] = sim  # Symmetric

    return matrix
