"""Pytest fixtures shared across all test modules.

Key Learning Points:
  - conftest.py is auto-discovered by pytest — fixtures here are available everywhere
  - Fixtures with 'session' scope are created once and reused across all tests
  - Mock providers let tests run without Ollama — fast, deterministic, offline
"""

import pytest

from rag_playground.config.env import settings
from rag_playground.shared.types import Chunk, Document, SearchResult
from tests.helpers.mock_provider import MockProvider


@pytest.fixture(scope="session")
def mock_provider() -> MockProvider:
    """A mock ModelProvider that returns deterministic responses.

    Use this for unit tests that need a provider but shouldn't
    call a real Ollama server.
    """
    return MockProvider()


@pytest.fixture(scope="session")
def sample_documents() -> list[Document]:
    """A small set of sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a field of artificial intelligence.",
            metadata={"source": "ml.md", "category": "tech"},
        ),
        Document(
            page_content="Climate change refers to long-term shifts in temperatures.",
            metadata={"source": "climate.md", "category": "science"},
        ),
        Document(
            page_content="World War II was a global conflict from 1939 to 1945.",
            metadata={"source": "ww2.md", "category": "history"},
        ),
        Document(
            page_content="The Renaissance was a period of European cultural rebirth.",
            metadata={"source": "renaissance.md", "category": "art"},
        ),
    ]


@pytest.fixture(scope="session")
def sample_chunks(sample_documents: list[Document]) -> list[Chunk]:
    """Chunks from sample documents."""
    chunks: list[Chunk] = []
    for doc in sample_documents:
        chunks.append(
            Chunk(
                id=f"{doc.metadata['source']}-0",
                content=doc.page_content,
                document_id=doc.metadata["source"],
                metadata=dict(doc.metadata),
                chunk_index=0,
            )
        )
    return chunks


@pytest.fixture(scope="session")
def sample_search_results(sample_chunks: list[Chunk]) -> list[SearchResult]:
    """Sample search results with decreasing similarity scores."""
    return [
        SearchResult(chunk=sample_chunks[0], score=0.95, rank=1),
        SearchResult(chunk=sample_chunks[1], score=0.82, rank=2),
        SearchResult(chunk=sample_chunks[2], score=0.71, rank=3),
        SearchResult(chunk=sample_chunks[3], score=0.45, rank=4),
    ]
