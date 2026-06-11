"""Tests for embedding generation (using mock provider)."""

from tests.helpers.mock_provider import MockProvider


class TestEmbedQuery:
    async def test_single_embed(self):
        provider = MockProvider()
        vector = await provider.embed_single("mock-model", "hello world")
        assert len(vector) == 4  # Mock produces 4-dim vectors
        assert all(isinstance(v, float) for v in vector)

    async def test_embed_documents_batch(self):
        provider = MockProvider()
        texts = ["hello", "world", "test"]
        result = await provider.embed("mock-model", texts)
        assert len(result.vectors) == 3
        assert all(len(v) == 4 for v in result.vectors)

    async def test_embed_empty_batch(self):
        provider = MockProvider()
        result = await provider.embed("mock-model", [])
        assert result.vectors == []
