"""Smoke tests for the config layer and shared types."""

from rag_playground.config.env import settings
from rag_playground.config.models import resolve_chat_config, resolve_embed_config
from rag_playground.config.providers.ollama import OllamaProvider
from rag_playground.shared.types import Chunk, Document, SearchResult
from tests.helpers.mock_provider import MockProvider


class TestSettings:
    """Verify default settings load correctly."""

    def test_default_chat_model(self):
        assert settings.chat_model == "qwen2.5:7b"

    def test_default_embed_model(self):
        assert settings.embed_model == "nomic-embed-text"

    def test_default_dimensions(self):
        assert settings.embed_dimensions == 768

    def test_default_top_k(self):
        assert settings.top_k == 3


class TestModelRegistry:
    """Verify model registry resolves providers correctly."""

    def test_resolve_chat_config(self):
        provider, model = resolve_chat_config()
        assert model == "qwen2.5:7b"
        assert isinstance(provider, OllamaProvider)

    def test_resolve_embed_config(self):
        provider, model, dims = resolve_embed_config()
        assert model == "nomic-embed-text"
        assert dims == 768
        assert isinstance(provider, OllamaProvider)

    def test_registry_is_cached(self):
        """lru_cache should return the same instance."""
        p1, _ = resolve_chat_config()
        p2, _ = resolve_chat_config()
        assert p1 is p2


class TestSharedTypes:
    """Verify core dataclasses work correctly."""

    def test_document_creation(self):
        doc = Document(
            page_content="Test content",
            metadata={"source": "test.md"},
        )
        assert doc.page_content == "Test content"
        assert doc.metadata["source"] == "test.md"

    def test_chunk_has_unique_id(self):
        c1 = Chunk(content="a", document_id="d1", chunk_index=0)
        c2 = Chunk(content="b", document_id="d1", chunk_index=1)
        assert c1.id != c2.id
        assert c1.chunk_index == 0
        assert c2.chunk_index == 1

    def test_search_result_ranking(self):
        chunk = Chunk(content="test", document_id="d1")
        result = SearchResult(chunk=chunk, score=0.85, rank=1)
        assert result.score == 0.85
        assert result.rank == 1
        assert result.chunk is chunk


class TestMockProvider:
    """Verify the mock provider works for testing."""

    async def test_mock_chat(self, mock_provider: MockProvider):
        from rag_playground.config.providers.base import ChatMessage

        msgs = [ChatMessage(role="user", content="Tell me about machine learning")]
        response = await mock_provider.chat("mock-model", msgs)
        assert "machine learning" in response.lower()

    async def test_mock_embed(self, mock_provider: MockProvider):
        result = await mock_provider.embed("mock-model", ["hello world"])
        assert len(result.vectors) == 1
        assert len(result.vectors[0]) == 4  # Mock produces 4-dim vectors

    async def test_mock_health_check(self, mock_provider: MockProvider):
        assert await mock_provider.health_check() is True


class TestFixtures:
    """Verify pytest fixtures are correctly set up."""

    def test_sample_documents(self, sample_documents: list[Document]):
        assert len(sample_documents) == 4
        assert all(isinstance(d, Document) for d in sample_documents)

    def test_sample_chunks(self, sample_chunks: list[Chunk]):
        assert len(sample_chunks) == 4
        assert all(isinstance(c, Chunk) for c in sample_chunks)
        assert all(c.content for c in sample_chunks)

    def test_sample_search_results(self, sample_search_results: list[SearchResult]):
        assert len(sample_search_results) == 4
        # Should be in descending order
        for i in range(len(sample_search_results) - 1):
            assert sample_search_results[i].score >= sample_search_results[i + 1].score
