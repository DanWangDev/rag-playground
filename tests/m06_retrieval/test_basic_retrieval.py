"""Tests for basic retrieval (using mock provider + in-memory store)."""

from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.m06_retrieval.basic_retrieval import retrieve
from rag_playground.shared.types import Chunk
from tests.helpers.mock_provider import MockProvider


class TestBasicRetrieval:
    async def test_retrieve_returns_results(self, mock_provider: MockProvider):
        store = InMemoryVectorStore(dimensions=4)
        for i in range(5):
            chunk = Chunk(
                id=f"c{i}",
                content=f"Content {i}",
                document_id=f"doc-{i}",
                metadata={"topic": "test"},
                chunk_index=i,
            )
            vec = [0.0] * 4
            vec[i % 4] = 1.0
            store.add(vec, chunk)

        result = await retrieve(
            "test query",
            store,
            top_k=3,
            embed_provider=mock_provider,
            embed_model="mock-model",
        )
        assert len(result.results) == 3
        assert result.strategy == "basic"
        assert result.elapsed_ms > 0

    async def test_retrieve_metadata_filter(self, mock_provider: MockProvider):
        store = InMemoryVectorStore(dimensions=4)
        for i, topic in enumerate(["ml", "ml", "climate"]):
            chunk = Chunk(
                id=f"c{i}",
                content=f"Content {i}",
                document_id=f"doc-{i}",
                metadata={"topic": topic},
                chunk_index=i,
            )
            vec = [0.0] * 4
            vec[i % 4] = 1.0
            store.add(vec, chunk)

        result = await retrieve(
            "test",
            store,
            top_k=10,
            metadata_filter={"topic": "ml"},
            embed_provider=mock_provider,
            embed_model="mock-model",
        )
        assert len(result.results) == 2
