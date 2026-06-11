"""In-memory vector store with brute-force k-NN search.

Key Learning Points:
  - Parallel arrays store vectors, chunks, and metadata
  - Brute-force cosine similarity search — O(n*d)
  - Metadata filtering post-search
  - Dimension enforcement on add()
"""

from rag_playground.m04_embeddings.similarity import cosine_similarity
from rag_playground.shared.types import Chunk, SearchResult


class InMemoryVectorStore:
    """A pure in-memory vector store with brute-force k-NN search.

    All data lives in Python lists. Fast for up to ~10K vectors.
    For larger collections, use LanceDB, Qdrant, or Pinecone.

    Attributes:
        embedding_model: Name of the embedding model used.
        dimensions: Expected vector dimensions (enforced on add).
    """

    def __init__(
        self, embedding_model: str = "nomic-embed-text", dimensions: int = 768
    ):
        self.embedding_model = embedding_model
        self.dimensions = dimensions

        self._vectors: list[list[float]] = []
        self._chunks: list[Chunk] = []
        self._metadata: list[dict] = []
        self._id_index: dict[str, int] = {}  # chunk_id → list position

    # ── Properties ──

    @property
    def size(self) -> int:
        """Number of entries in the store."""
        return len(self._vectors)

    @property
    def is_empty(self) -> bool:
        """True if the store has no entries."""
        return self.size == 0

    # ── CRUD ──

    def add(
        self, vector: list[float], chunk: Chunk, metadata: dict | None = None
    ) -> int:
        """Add a vector-chunk pair to the store.

        Args:
            vector: The embedding vector.
            chunk: The chunk this vector represents.
            metadata: Optional additional metadata.

        Returns:
            The index position where this entry was stored.

        Raises:
            ValueError: If vector dimensions don't match the store.
        """
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimensions}, got {len(vector)}"
            )

        idx = self.size
        self._vectors.append(vector)
        self._chunks.append(chunk)
        self._metadata.append(metadata or {})
        self._id_index[chunk.id] = idx

        return idx

    def add_batch(
        self, entries: list[tuple[list[float], Chunk, dict | None]]
    ) -> list[int]:
        """Add multiple vector-chunk pairs at once.

        Args:
            entries: List of (vector, chunk, metadata) tuples.

        Returns:
            List of index positions.
        """
        return [self.add(vec, chunk, meta) for vec, chunk, meta in entries]

    def get_by_id(self, chunk_id: str) -> tuple[list[float], Chunk, dict] | None:
        """Look up an entry by its chunk ID."""
        idx = self._id_index.get(chunk_id)
        if idx is None:
            return None
        return (self._vectors[idx], self._chunks[idx], self._metadata[idx])

    def delete(self, chunk_id: str) -> bool:
        """Delete an entry by chunk ID. Returns True if found and deleted.

        Note: Deleting from a parallel-array store is O(n) because we
        rebuild all arrays. For frequent deletes, consider a different structure.
        """
        idx = self._id_index.pop(chunk_id, None)
        if idx is None:
            return False

        del self._vectors[idx]
        del self._chunks[idx]
        del self._metadata[idx]

        # Rebuild the ID index (indices shifted after deletion)
        self._id_index = {}
        for i, chunk in enumerate(self._chunks):
            self._id_index[chunk.id] = i

        return True

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._vectors.clear()
        self._chunks.clear()
        self._metadata.clear()
        self._id_index.clear()

    # ── Search ──

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        min_score: float | None = None,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for the top-k most similar vectors.

        Brute-force O(n*d) cosine similarity scan.

        Args:
            query_vector: The query embedding.
            top_k: Number of results to return.
            min_score: Minimum similarity threshold (0.0 to 1.0).
            metadata_filter: Only return results whose metadata matches all keys in this dict.

        Returns:
            Ranked list of SearchResult objects.

        Raises:
            ValueError: If the store is empty.
        """
        if self.is_empty:
            raise ValueError("Cannot search an empty vector store")

        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimensions}, got {len(query_vector)}"
            )

        # Compute similarity for all vectors
        scored: list[SearchResult] = []
        for i, vec in enumerate(self._vectors):
            # Apply metadata pre-filter
            if metadata_filter:
                if not self._matches_filter(
                    self._chunks[i].metadata, self._metadata[i], metadata_filter
                ):
                    continue

            sim = cosine_similarity(query_vector, vec)

            # Apply score threshold
            if min_score is not None and sim < min_score:
                continue

            scored.append(
                SearchResult(
                    chunk=self._chunks[i],
                    score=sim,
                    rank=0,  # Will be set after sorting
                )
            )

        # Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)

        # Assign ranks and limit to top_k
        for rank, result in enumerate(scored[:top_k], 1):
            result.rank = rank

        return scored[:top_k]

    @staticmethod
    def _matches_filter(chunk_meta: dict, store_meta: dict, filter_dict: dict) -> bool:
        """Check if metadata matches all filter conditions.

        Searches both chunk metadata and per-entry store metadata.
        """
        merged = {**chunk_meta, **store_meta}
        for key, value in filter_dict.items():
            if key not in merged or merged[key] != value:
                return False
        return True

    # ── Serialization ──

    def to_dict(self) -> dict:
        """Serialize the store to a dictionary (for JSON export)."""
        return {
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "entries": [
                {
                    "vector": vec,
                    "chunk": {
                        "id": chunk.id,
                        "content": chunk.content,
                        "document_id": chunk.document_id,
                        "metadata": chunk.metadata,
                        "chunk_index": chunk.chunk_index,
                    },
                    "metadata": meta,
                }
                for vec, chunk, meta in zip(self._vectors, self._chunks, self._metadata)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InMemoryVectorStore":
        """Deserialize a store from a dictionary."""
        store = cls(
            embedding_model=data["embedding_model"],
            dimensions=data["dimensions"],
        )

        for entry in data["entries"]:
            chunk_data = entry["chunk"]
            chunk = Chunk(
                id=chunk_data["id"],
                content=chunk_data["content"],
                document_id=chunk_data["document_id"],
                metadata=chunk_data.get("metadata", {}),
                chunk_index=chunk_data.get("chunk_index", 0),
            )
            store.add(
                vector=entry["vector"],
                chunk=chunk,
                metadata=entry.get("metadata", {}),
            )

        return store
