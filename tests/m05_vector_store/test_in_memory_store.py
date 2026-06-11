"""Tests for InMemoryVectorStore."""

import pytest

from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.shared.types import Chunk


class TestInMemoryVectorStore:
    def _make_chunk(self, idx: int, topic: str = "test") -> Chunk:
        return Chunk(
            id=f"chunk-{idx}",
            content=f"Content {idx}",
            document_id=f"doc-{idx}",
            metadata={"topic": topic},
            chunk_index=idx,
        )

    def test_add_and_size(self):
        store = InMemoryVectorStore(dimensions=4)
        chunk = self._make_chunk(0)
        store.add([1.0, 0.0, 0.0, 0.0], chunk)
        assert store.size == 1

    def test_dimension_mismatch_raises(self):
        store = InMemoryVectorStore(dimensions=4)
        with pytest.raises(ValueError):
            store.add([1.0, 2.0], self._make_chunk(0))

    def test_search_returns_ordered_results(self):
        store = InMemoryVectorStore(dimensions=2)
        store.add([1.0, 0.0], self._make_chunk(0, "ml"))
        store.add([0.0, 1.0], self._make_chunk(1, "climate"))
        store.add([0.9, 0.1], self._make_chunk(2, "ml"))

        results = store.search([1.0, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0].score > results[1].score >= results[2].score

    def test_search_top_k_limits(self):
        store = InMemoryVectorStore(dimensions=2)
        for i in range(5):
            store.add([float(i), 0.0], self._make_chunk(i))

        results = store.search([0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_search_empty_store_raises(self):
        store = InMemoryVectorStore(dimensions=2)
        with pytest.raises(ValueError):
            store.search([1.0, 0.0])

    def test_min_score_filter(self):
        store = InMemoryVectorStore(dimensions=2)
        store.add([1.0, 0.0], self._make_chunk(0))  # Perfect match
        store.add([0.0, 1.0], self._make_chunk(1))  # Orthogonal

        results = store.search([1.0, 0.0], top_k=10, min_score=0.5)
        assert len(results) == 1  # Only the perfect match

    def test_metadata_filter(self):
        store = InMemoryVectorStore(dimensions=2)
        store.add([1.0, 0.0], self._make_chunk(0, "ml"))
        store.add([0.9, 0.1], self._make_chunk(1, "ml"))
        store.add([0.0, 1.0], self._make_chunk(2, "climate"))

        results = store.search([1.0, 0.0], top_k=10, metadata_filter={"topic": "ml"})
        assert len(results) == 2
        assert all(r.chunk.metadata["topic"] == "ml" for r in results)

    def test_add_batch(self):
        store = InMemoryVectorStore(dimensions=2)
        entries = [
            ([1.0, 0.0], self._make_chunk(0), {"idx": 0}),
            ([0.0, 1.0], self._make_chunk(1), {"idx": 1}),
        ]
        indices = store.add_batch(entries)
        assert indices == [0, 1]
        assert store.size == 2

    def test_get_by_id(self):
        store = InMemoryVectorStore(dimensions=2)
        chunk = self._make_chunk(42)
        store.add([1.0, 0.0], chunk)
        result = store.get_by_id(chunk.id)
        assert result is not None
        assert result[1].id == chunk.id

    def test_delete(self):
        store = InMemoryVectorStore(dimensions=2)
        chunk = self._make_chunk(0)
        store.add([1.0, 0.0], chunk)
        assert store.size == 1
        assert store.delete(chunk.id)
        assert store.size == 0

    def test_clear(self):
        store = InMemoryVectorStore(dimensions=2)
        store.add([1.0, 0.0], self._make_chunk(0))
        store.add([0.0, 1.0], self._make_chunk(1))
        store.clear()
        assert store.size == 0
        assert store.is_empty
