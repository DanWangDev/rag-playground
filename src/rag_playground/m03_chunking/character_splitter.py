"""Fixed-size character splitting.

Key Learning Points:
  - Simplest splitting strategy — cut at every N characters
  - Fast and deterministic, but can cut mid-word or mid-sentence
  - Overlap ensures key information isn't lost at chunk boundaries
  - Chunk metadata tracks position within parent document
"""

from rag_playground.m02_data_loading.document import Document
from rag_playground.shared.types import Chunk


def split_by_characters(
    doc: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a document into fixed-size character chunks.

    Args:
        doc: The document to split.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of Chunk objects, each a slice of the original document.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    text = doc.page_content
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        # Skip empty or whitespace-only chunks
        if chunk_text.strip():
            chunk = Chunk(
                id=f"{doc.metadata.get('source', 'unknown')}-chunk-{index}",
                content=chunk_text,
                document_id=doc.metadata.get("source", ""),
                metadata={
                    **doc.metadata,
                    "chunk_index": index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_strategy": "character",
                },
                chunk_index=index,
            )
            chunks.append(chunk)
            index += 1

        # Move forward by (chunk_size - overlap) for the next chunk
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1  # Prevent infinite loop
        start += step

    return chunks
