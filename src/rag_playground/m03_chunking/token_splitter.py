"""Token-aware text splitting.

Key Learning Points:
  - Tokens ≠ characters — ~4 chars per token in English, more in other languages
  - Simple approximation: split on whitespace, count "words" as token proxies
  - Real tokenizers (tiktoken, HuggingFace) are more accurate but require deps
  - This module teaches the concept without external tokenizer libraries
"""

from rag_playground.m02_data_loading.document import Document
from rag_playground.shared.types import Chunk


def split_by_tokens(
    doc: Document,
    chunk_tokens: int = 128,
    chunk_overlap_tokens: int = 16,
    chars_per_token: float = 4.0,
) -> list[Chunk]:
    """Split a document into token-count-aware chunks.

    Uses a simple heuristic: average characters per token to estimate
    token counts without a real tokenizer.

    Args:
        doc: The document to split.
        chunk_tokens: Target tokens per chunk.
        chunk_overlap_tokens: Token overlap between chunks.
        chars_per_token: Estimated characters per token (English ≈ 4).

    Returns:
        List of Chunk objects with approximate token counts in metadata.
    """
    text = doc.page_content
    if not text:
        return []

    # Word-level splitting gives better token alignment than character-level.
    # Token budget is tracked per-word (approximated as len(word) / 4).
    words = text.split()
    chunks: list[Chunk] = []
    current_words: list[str] = []
    current_tokens = 0
    index = 0

    for word in words:
        # Approximate tokens for this word: ceil(len / 4)
        word_tokens = max(1, len(word) // 4)

        if current_tokens + word_tokens > chunk_tokens and current_words:
            # Build chunk from accumulated words
            chunk_text = " ".join(current_words)
            approx_tokens = sum(max(1, len(w) // 4) for w in current_words)

            chunks.append(
                Chunk(
                    id=f"{doc.metadata.get('source', 'unknown')}-chunk-{index}",
                    content=chunk_text,
                    document_id=doc.metadata.get("source", ""),
                    metadata={
                        **doc.metadata,
                        "chunk_index": index,
                        "chunk_strategy": "token",
                        "approx_tokens": approx_tokens,
                        "chars_per_token_estimate": chars_per_token,
                    },
                    chunk_index=index,
                )
            )
            index += 1

            # Start new chunk with overlap
            if chunk_overlap_tokens > 0:
                # Keep last N words for overlap
                overlap_words_count = max(1, chunk_overlap_tokens // 2)
                overlap_words = current_words[-overlap_words_count:]
                current_words = overlap_words
                current_tokens = sum(max(1, len(w) // 4) for w in overlap_words)
            else:
                current_words = []
                current_tokens = 0

        current_words.append(word)
        current_tokens += word_tokens

    # Last chunk
    if current_words:
        chunk_text = " ".join(current_words)
        approx_tokens = sum(max(1, len(w) // 4) for w in current_words)

        chunks.append(
            Chunk(
                id=f"{doc.metadata.get('source', 'unknown')}-chunk-{index}",
                content=chunk_text,
                document_id=doc.metadata.get("source", ""),
                metadata={
                    **doc.metadata,
                    "chunk_index": index,
                    "chunk_strategy": "token",
                    "approx_tokens": approx_tokens,
                    "chars_per_token_estimate": chars_per_token,
                },
                chunk_index=index,
            )
        )

    return chunks
