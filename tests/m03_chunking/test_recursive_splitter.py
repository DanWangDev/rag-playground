"""Tests for recursive splitter."""

from rag_playground.m02_data_loading.document import Document
from rag_playground.m03_chunking.recursive_splitter import split_recursively
from rag_playground.shared.types import Chunk


class TestSplitRecursively:
    def test_basic_split(self):
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = Document(page_content=content, metadata={"source": "test.md"})
        chunks = split_recursively(doc, chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_splits_on_paragraphs(self):
        content = "First paragraph with enough text to matter.\n\nSecond paragraph also has enough text."
        doc = Document(page_content=content, metadata={"source": "test.md"})
        # Small chunk size forces splitting
        chunks = split_recursively(doc, chunk_size=40, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_no_split_needed(self):
        content = "Short document."
        doc = Document(page_content=content, metadata={"source": "test.md"})
        chunks = split_recursively(doc, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1
        assert "Short document." in chunks[0].content

    def test_empty_content(self):
        doc = Document(page_content="", metadata={"source": "test.md"})
        chunks = split_recursively(doc)
        assert chunks == []

    def test_metadata_preserved(self):
        content = "Sentence one. Sentence two. Sentence three. Sentence four. " * 5
        doc = Document(
            page_content=content,
            metadata={"source": "test.md", "category": "test"},
        )
        chunks = split_recursively(doc, chunk_size=80, chunk_overlap=0)
        for c in chunks:
            assert c.metadata["source"] == "test.md"
            assert c.metadata["category"] == "test"
            assert c.metadata["chunk_strategy"] == "recursive"

    def test_custom_separators(self):
        # Long words that won't fit in one chunk with chunk_size=20
        content = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        doc = Document(page_content=content, metadata={"source": "test.md"})
        # Use space as primary separator — each word is ~5 chars
        chunks = split_recursively(
            doc, chunk_size=20, chunk_overlap=0, separators=[" ", ""]
        )
        assert len(chunks) >= 2
