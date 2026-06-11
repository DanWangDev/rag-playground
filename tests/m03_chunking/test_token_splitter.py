"""Tests for token splitter."""

from rag_playground.m02_data_loading.document import Document
from rag_playground.m03_chunking.token_splitter import split_by_tokens
from rag_playground.shared.types import Chunk


class TestSplitByTokens:
    def test_basic_split(self):
        words = "the quick brown fox " * 50
        doc = Document(page_content=words.strip(), metadata={"source": "test.txt"})
        chunks = split_by_tokens(doc, chunk_tokens=20, chunk_overlap_tokens=0)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_single_chunk(self):
        doc = Document(
            page_content="just a few words here", metadata={"source": "test.txt"}
        )
        chunks = split_by_tokens(doc, chunk_tokens=100)
        assert len(chunks) == 1

    def test_empty_content(self):
        doc = Document(page_content="", metadata={"source": "test.txt"})
        chunks = split_by_tokens(doc)
        assert chunks == []

    def test_token_metadata(self):
        words = "hello world this is a test document for token counting " * 10
        doc = Document(page_content=words.strip(), metadata={"source": "test.txt"})
        chunks = split_by_tokens(doc, chunk_tokens=15, chunk_overlap_tokens=0)

        for c in chunks:
            assert "approx_tokens" in c.metadata
            assert c.metadata["chunk_strategy"] == "token"
            assert c.metadata["chars_per_token_estimate"] == 4.0

    def test_overlap(self):
        words = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 5
        doc = Document(page_content=words.strip(), metadata={"source": "test.txt"})
        no_overlap = split_by_tokens(doc, chunk_tokens=10, chunk_overlap_tokens=0)
        with_overlap = split_by_tokens(doc, chunk_tokens=10, chunk_overlap_tokens=4)
        # Overlap should produce more chunks (or same number)
        assert len(with_overlap) >= len(no_overlap)

    def test_custom_chars_per_token(self):
        words = "hello world test document " * 10
        doc = Document(page_content=words.strip(), metadata={"source": "test.txt"})
        chunks = split_by_tokens(doc, chunk_tokens=10, chars_per_token=2.0)
        # With 2 chars/token, chunk_size_chars = 20 chars. Each chunk should be small.
        assert all(len(c.content) > 0 for c in chunks)
        assert len(chunks) >= 2  # Should produce multiple small chunks
