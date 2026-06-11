"""Recursive character splitting with separator hierarchy.

Key Learning Points:
  - Tries to split on natural boundaries before falling back to character splits
  - Separator hierarchy: paragraphs → lines → sentences → words → characters
  - Produces semantically coherent chunks (rarely cuts mid-word)
  - The recursive approach mirrors LangChain's RecursiveCharacterTextSplitter
"""

from rag_playground.m02_data_loading.document import Document
from rag_playground.shared.types import Chunk

# Separator hierarchy: try the most semantic boundary first
DEFAULT_SEPARATORS = [
    "\n\n",  # Paragraphs
    "\n",  # Lines
    ". ",  # Sentences
    "? ",  # Questions
    "! ",  # Exclamations
    "; ",  # Clause boundaries
    ", ",  # Phrase boundaries
    " ",  # Words
    "",  # Characters (last resort)
]


def split_recursively(
    doc: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Split a document using recursive separator hierarchy.

    Tries each separator in order. For each separator:
      1. Split the text on that separator
      2. Merge splits that are small enough to fit in one chunk
      3. Recurse into splits that are still too large with the next separator

    Args:
        doc: The document to split.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.
        separators: Hierarchy of separators (default: paragraphs → lines → sentences → words → chars).

    Returns:
        List of Chunk objects.
    """
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    text = doc.page_content

    if not text:
        return []

    splits = _split_text(text, seps, chunk_size)
    return _merge_splits(splits, doc, chunk_size, chunk_overlap, seps)


def _split_text(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text into pieces no larger than chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator
    for sep in separators:
        if sep == "":
            # Character-level: just split evenly
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        if sep in text:
            pieces = text.split(sep)
            # Recurse into pieces that are too large
            result: list[str] = []
            current = ""
            for piece in pieces:
                # Add separator back (except for the last piece)
                candidate = current + piece if not current else current + sep + piece
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        result.append(current)
                    # Recurse into this piece with remaining separators
                    remaining_seps = separators[separators.index(sep) + 1 :]
                    sub_splits = _split_text(piece, remaining_seps, chunk_size)
                    result.extend(sub_splits)
                    current = ""
            if current.strip():
                result.append(current)
            return result

    # No separator matched — return as single piece
    return [text] if text.strip() else []


def _merge_splits(
    splits: list[str],
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[Chunk]:
    """Merge splits into chunks with overlap, tracking metadata."""
    chunks: list[Chunk] = []
    current_chunk = ""
    index = 0

    for split in splits:
        candidate = current_chunk + (" " if current_chunk else "") + split
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(
                    Chunk(
                        id=f"{doc.metadata.get('source', 'unknown')}-chunk-{index}",
                        content=current_chunk.strip(),
                        document_id=doc.metadata.get("source", ""),
                        metadata={
                            **doc.metadata,
                            "chunk_index": index,
                            "chunk_strategy": "recursive",
                        },
                        chunk_index=index,
                    )
                )
                index += 1

            # Start new chunk with overlap from previous
            if chunk_overlap > 0 and current_chunk:
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                overlap_text = current_chunk[overlap_start:]
                current_chunk = overlap_text + " " + split if overlap_text else split
            else:
                current_chunk = split

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(
            Chunk(
                id=f"{doc.metadata.get('source', 'unknown')}-chunk-{index}",
                content=current_chunk.strip(),
                document_id=doc.metadata.get("source", ""),
                metadata={
                    **doc.metadata,
                    "chunk_index": index,
                    "chunk_strategy": "recursive",
                },
                chunk_index=index,
            )
        )

    return chunks
