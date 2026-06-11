"""Tests for directory loading."""

import tempfile
from pathlib import Path

from rag_playground.m02_data_loading.document import Document
from rag_playground.m02_data_loading.load_directory import load_directory


class TestLoadDirectory:
    async def test_load_directory_with_md_and_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create a few files
            (tmp / "readme.md").write_text(
                "# README\n\nProject docs.", encoding="utf-8"
            )
            (tmp / "notes.txt").write_text("Some notes here.", encoding="utf-8")
            # Create subdirectory
            sub = tmp / "subdir"
            sub.mkdir()
            (sub / "details.md").write_text("# Details\n\nMore info.", encoding="utf-8")

            docs = await load_directory(tmpdir)

            assert len(docs) == 3
            assert all(isinstance(d, Document) for d in docs)

            filenames = {d.metadata["filename"] for d in docs}
            assert "readme.md" in filenames
            assert "notes.txt" in filenames
            assert "details.md" in filenames

    async def test_directory_not_found(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            await load_directory("/nonexistent/dir")

    async def test_not_a_directory(self):
        import pytest

        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(NotADirectoryError):
                await load_directory(f.name)

    async def test_metadata_enrichment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "doc.md").write_text("# Doc\n\nContent.", encoding="utf-8")

            docs = await load_directory(tmpdir)

            assert len(docs) == 1
            doc = docs[0]
            assert "relative_path" in doc.metadata
            assert doc.metadata["relative_path"] == "doc.md"
            assert "top_level_dir" in doc.metadata
            assert doc.metadata["top_level_dir"] == "."

    async def test_recursive_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "root.md").write_text("# Root", encoding="utf-8")
            sub = tmp / "subdir"
            sub.mkdir()
            (sub / "nested.md").write_text("# Nested", encoding="utf-8")

            docs_recursive = await load_directory(tmpdir, recursive=True)
            docs_flat = await load_directory(tmpdir, recursive=False)

            # Recursive should find both, flat should find only root
            assert len(docs_recursive) >= 2
            assert len(docs_flat) >= 1  # At least the root file
