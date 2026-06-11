"""Custom assertion helpers for RAG Playground tests.

Key Learning Points:
  - Custom assertions make test intent clearer than raw numpy comparisons
  - Floating-point comparisons always use tolerance (abs_tol)
  - Assertion helpers centralize common check patterns across test modules
"""

import math


def assert_valid_embedding(vector: list[float], expected_dimensions: int = 0) -> None:
    """Assert that a vector looks like a valid embedding.

    Checks:
      - Not empty
      - Correct dimensions (if specified)
      - No NaN or infinity values
      - Normalized to approximately unit length
    """
    assert len(vector) > 0, "Embedding vector should not be empty"

    if expected_dimensions > 0:
        assert len(vector) == expected_dimensions, (
            f"Expected {expected_dimensions} dimensions, got {len(vector)}"
        )

    for v in vector:
        assert not math.isnan(v), "Embedding contains NaN"
        assert not math.isinf(v), "Embedding contains infinity"

    # Check approximate unit length (real embeddings are typically normalized)
    norm = math.sqrt(sum(v * v for v in vector))
    assert 0.1 < norm < 10.0, f"Vector norm {norm:.4f} is outside expected range"


def assert_similarity_between(
    score: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> None:
    """Assert a similarity score is within the expected range."""
    assert min_val <= score <= max_val, (
        f"Similarity {score:.4f} not in range [{min_val}, {max_val}]"
    )


def assert_vector_close(
    a: list[float],
    b: list[float],
    abs_tol: float = 1e-6,
) -> None:
    """Assert two vectors are element-wise close within tolerance."""
    assert len(a) == len(b), f"Vector lengths differ: {len(a)} vs {len(b)}"
    for i, (va, vb) in enumerate(zip(a, b)):
        assert abs(va - vb) < abs_tol, f"Vectors differ at index {i}: {va} vs {vb}"


def assert_chunk_valid(chunk) -> None:
    """Assert a chunk has all required fields with valid values."""
    assert chunk.id, "Chunk should have an id"
    assert chunk.content, "Chunk should have content"
    assert chunk.document_id, "Chunk should have a document_id"
    assert chunk.chunk_index >= 0, (
        f"chunk_index should be >= 0, got {chunk.chunk_index}"
    )


def assert_search_results_ordered(results: list) -> None:
    """Assert search results are in descending score order."""
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, (
            f"Results not ordered: rank {results[i].rank} ({results[i].score}) < rank {results[i + 1].rank} ({results[i + 1].score})"
        )
