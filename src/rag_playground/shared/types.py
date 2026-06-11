"""Shared type definitions for RAG Playground.

Core dataclasses used across all modules. These define the shape of data
as it flows through the RAG pipeline: Document → Chunk → Embedding → SearchResult.

Key Learning Points:
  - Dataclasses provide immutable-ish data containers with minimal boilerplate
  - Each stage of the RAG pipeline has its own data type
  - Metadata dicts carry provenance (where did this chunk come from?)
  - IDs enable traceability from final answer back to source document
"""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Document:
    """A raw document loaded from disk.

    Mirrors LangChain's Document interface for familiarity, but
    implemented from scratch.

    Attributes:
        page_content: The full text content of the document.
        metadata: Arbitrary key-value metadata (source file, title, etc.).
    """

    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A segment of a document, created by a text splitter.

    Attributes:
        id: Unique identifier for this chunk.
        content: The chunk's text content.
        document_id: Which document this chunk came from.
        metadata: Metadata merged from the parent document + chunk position.
        chunk_index: Position of this chunk within the document (0-based).
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    document_id: str = ""
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


@dataclass
class EmbeddingVector:
    """A vector embedding with its source chunk reference.

    Attributes:
        chunk_id: The chunk this embedding represents.
        vector: The embedding vector (list of floats).
        model: Which embedding model produced this.
        dimensions: Length of the vector.
    """

    chunk_id: str
    vector: list[float]
    model: str = ""
    dimensions: int = 0


@dataclass
class SearchResult:
    """A single search result from vector similarity search.

    Attributes:
        chunk: The retrieved chunk.
        score: Similarity score (higher = more relevant, range depends on metric).
        rank: Position in search results (1-based).
    """

    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class RetrievalResult:
    """Complete result from a retrieval operation.

    Attributes:
        query: The original query text.
        results: Ranked list of search results.
        strategy: Which retrieval strategy was used.
        elapsed_ms: Time taken for retrieval in milliseconds.
    """

    query: str
    results: list[SearchResult]
    strategy: str = "basic"
    elapsed_ms: float = 0.0


@dataclass
class RAGResponse:
    """Complete response from a RAG query.

    Attributes:
        answer: The generated answer text.
        sources: The chunks that were retrieved and used for context.
        prompt_tokens: Estimated tokens in the prompt.
        completion_tokens: Estimated tokens in the generated answer.
        elapsed_ms: Total time from query to answer.
    """

    answer: str
    sources: list[SearchResult]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
