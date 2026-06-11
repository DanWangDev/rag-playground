"""Tests for cosine similarity and nearest-neighbor search."""

import math

import pytest

from rag_playground.m04_embeddings.similarity import (
    cosine_similarity,
    find_most_similar,
    similarity_matrix,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        sim = cosine_similarity(v, v)
        assert math.isclose(sim, 1.0, rel_tol=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = cosine_similarity(a, b)
        assert math.isclose(sim, 0.0, abs_tol=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        sim = cosine_similarity(a, b)
        assert math.isclose(sim, -1.0, rel_tol=1e-6)

    def test_similar_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        sim = cosine_similarity(a, b)
        assert sim > 0.99  # Very similar

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector(self):
        sim = cosine_similarity([0.0, 0.0], [1.0, 2.0])
        assert sim == 0.0


class TestFindMostSimilar:
    def test_top_1(self):
        query = [1.0, 0.0]
        candidates = [
            [1.0, 0.0],  # Perfect match
            [0.0, 1.0],  # Orthogonal
            [-1.0, 0.0],  # Opposite
        ]
        results = find_most_similar(query, candidates, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 0  # Index 0 is the perfect match
        assert math.isclose(results[0][1], 1.0, rel_tol=1e-6)

    def test_top_k_ordering(self):
        query = [1.0, 0.0, 0.0]
        candidates = [
            [0.9, 0.1, 0.0],  # Most similar
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],  # Least similar
        ]
        results = find_most_similar(query, candidates, top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self):
        results = find_most_similar([1.0, 0.0], [])
        assert results == []

    def test_top_k_exceeds_candidates(self):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        results = find_most_similar(query, candidates, top_k=10)
        assert len(results) == 2  # Only 2 candidates available


class TestSimilarityMatrix:
    def test_basic_matrix(self):
        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        matrix = similarity_matrix(vectors)
        assert len(matrix) == 3
        assert len(matrix[0]) == 3
        # Diagonal should be 1.0 (self-similarity)
        for i in range(3):
            assert math.isclose(matrix[i][i], 1.0, rel_tol=1e-6)
        # Matrix should be symmetric
        for i in range(3):
            for j in range(3):
                assert math.isclose(matrix[i][j], matrix[j][i], rel_tol=1e-6)
