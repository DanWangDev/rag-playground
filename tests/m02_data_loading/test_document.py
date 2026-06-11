"""Tests for document loading (text and markdown)."""

import tempfile
from pathlib import Path

import pytest

from rag_playground.m02_data_loading.document import (
    Document,
    load_markdown_file,
    load_text_file,
)


class TestLoadTextFile:
    async def test_load_basic_text(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello, world!")
            tmp_path = f.name

        try:
            doc = await load_text_file(tmp_path)
            assert isinstance(doc, Document)
            assert doc.page_content == "Hello, world!"
            assert doc.metadata["file_type"] == ".txt"
            assert doc.metadata["encoding"] == "utf-8"
            assert "source" in doc.metadata
            assert "file_size" in doc.metadata
        finally:
            Path(tmp_path).unlink()

    async def test_load_multiline_text(self):
        content = "Line 1\nLine 2\nLine 3\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            doc = await load_text_file(tmp_path)
            assert doc.page_content == content
            assert "Line 1" in doc.page_content
        finally:
            Path(tmp_path).unlink()

    async def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            await load_text_file("/nonexistent/file.txt")

    async def test_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(IsADirectoryError):
                await load_text_file(tmpdir)


class TestLoadMarkdownFile:
    async def test_load_basic_markdown(self):
        content = "# Title\n\nSome content."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            doc = await load_markdown_file(tmp_path)
            assert "# Title" in doc.page_content
            assert doc.metadata["file_type"] == ".md"
        finally:
            Path(tmp_path).unlink()

    async def test_load_with_frontmatter(self):
        content = "---\ntitle: Test Doc\nauthor: Jane\ntags: [python, rag]\n---\n\n# Hello\n\nThis is the body."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            doc = await load_markdown_file(tmp_path)
            assert doc.metadata["title"] == "Test Doc"
            assert doc.metadata["author"] == "Jane"
            assert doc.metadata["tags"] == ["python", "rag"]
            assert "# Hello" in doc.page_content
            assert "This is the body" in doc.page_content
            # Frontmatter should NOT be in page_content
            assert "---" not in doc.page_content
        finally:
            Path(tmp_path).unlink()

    async def test_load_no_frontmatter(self):
        content = "# Just a heading\n\nNo frontmatter here."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            doc = await load_markdown_file(tmp_path)
            assert "# Just a heading" in doc.page_content
            assert "title" not in doc.metadata  # No frontmatter parsed
        finally:
            Path(tmp_path).unlink()

    async def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            await load_markdown_file("/nonexistent/file.md")
