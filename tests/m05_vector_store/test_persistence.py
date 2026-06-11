"""Tests for vector store JSON persistence."""

import tempfile
from pathlib import Path

from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.m05_vector_store.persistence import load_from_file, save_to_file
from rag_playground.shared.types import Chunk


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        store = InMemoryVectorStore(embedding_model="test-model", dimensions=4)

        for i in range(3):
            chunk = Chunk(
                id=f"chunk-{i}",
                content=f"Content {i}",
                document_id=f"doc-{i}",
                metadata={"topic": "test"},
                chunk_index=i,
            )
            store.add([float(i)] * 4, chunk, {"added_at": f"2024-01-{15 + i}"})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            save_to_file(store, tmp_path)
            loaded = load_from_file(tmp_path)

            assert loaded.size == store.size
            assert loaded.embedding_model == store.embedding_model
            assert loaded.dimensions == store.dimensions

            # Verify search still works
            results = loaded.search([0.0, 0.0, 0.0, 0.0], top_k=2)
            assert len(results) == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_from_file("/nonexistent/store.json")
