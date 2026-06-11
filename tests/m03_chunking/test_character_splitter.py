"""Tests for character splitter."""

import pytest

from rag_playground.m02_data_loading.document import Document
from rag_playground.m03_chunking.character_splitter import split_by_characters
from rag_playground.shared.types import Chunk


class TestSplitByCharacters:
    def test_basic_split(self):
        doc = Document(
            page_content="A" * 1000,
            metadata={"source": "test.txt"},
        )
        chunks = split_by_characters(doc, chunk_size=200, chunk_overlap=0)
        assert len(chunks) == 5
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(len(c.content) == 200 for c in chunks)

    def test_overlap(self):
        doc = Document(page_content="A" * 300, metadata={"source": "test.txt"})
        chunks = split_by_characters(doc, chunk_size=100, chunk_overlap=20)
        # With 100 size and 20 overlap, step is 80
        # 300 chars: chunks at 0-100, 80-180, 160-260, 240-300
        assert len(chunks) == 4

    def test_empty_content(self):
        doc = Document(page_content="", metadata={"source": "test.txt"})
        chunks = split_by_characters(doc)
        assert chunks == []

    def test_chunk_smaller_than_size(self):
        doc = Document(page_content="Short text", metadata={"source": "test.txt"})
        chunks = split_by_characters(doc, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_metadata_preserved(self):
        doc = Document(
            page_content="Hello world " * 50,
            metadata={"source": "test.txt", "author": "Test"},
        )
        chunks = split_by_characters(doc, chunk_size=100, chunk_overlap=0)
        for c in chunks:
            assert c.metadata["source"] == "test.txt"
            assert c.metadata["author"] == "Test"
            assert "chunk_index" in c.metadata
            assert c.metadata["chunk_strategy"] == "character"

    def test_invalid_size(self):
        doc = Document(page_content="test")
        with pytest.raises(ValueError):
            split_by_characters(doc, chunk_size=0)

    def test_overlap_too_large(self):
        doc = Document(page_content="test")
        with pytest.raises(ValueError):
            split_by_characters(doc, chunk_size=100, chunk_overlap=100)
