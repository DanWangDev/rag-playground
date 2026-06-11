"""Tests for RAG pipeline using mock provider."""

import pytest

from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.m07_rag_pipeline.generate import generate_answer
from rag_playground.shared.types import Chunk
from tests.helpers.mock_provider import MockProvider


class TestRAGPipeline:
    async def test_generate_answer_returns_rag_response(
        self, mock_provider: MockProvider
    ):
        store = InMemoryVectorStore(dimensions=4)
        for i in range(3):
            chunk = Chunk(
                id=f"c{i}",
                content=f"Document {i}: This is about {'machine learning' if i < 2 else 'climate change'}.",
                document_id=f"doc-{i}",
                metadata={"topic": "test"},
                chunk_index=i,
            )
            vec = [0.0] * 4
            vec[i] = 1.0
            store.add(vec, chunk)

        response = await generate_answer(
            "What is machine learning?",
            store,
            top_k=2,
            embed_provider=mock_provider,
            embed_model="mock-model",
            chat_provider=mock_provider,
            chat_model="mock-model",
        )
        assert response.answer
        assert len(response.sources) > 0
        assert response.elapsed_ms > 0

    async def test_empty_store_raises(self, mock_provider: MockProvider):
        store = InMemoryVectorStore(dimensions=2)
        with pytest.raises(ValueError):
            await generate_answer(
                "test",
                store,
                embed_provider=mock_provider,
                embed_model="mock-model",
                chat_provider=mock_provider,
                chat_model="mock-model",
            )
