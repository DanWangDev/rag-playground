"""Interactive exercise: Text Splitting & Chunking.

Usage:
    python -m rag_playground.m03_chunking.exercise
    python -m rag_playground.m03_chunking.exercise --no-pause
"""

import argparse
import asyncio
import os

from rag_playground.config.env import settings
from rag_playground.m02_data_loading.load_directory import load_directory
from rag_playground.m03_chunking.character_splitter import split_by_characters
from rag_playground.m03_chunking.recursive_splitter import split_recursively
from rag_playground.m03_chunking.token_splitter import split_by_tokens
from rag_playground.shared.logger import (
    divider,
    info,
    result,
    section,
    step,
    success,
    table,
    warning,
)
from rag_playground.shared.prompt import wait_for_user


async def main() -> None:
    parser = argparse.ArgumentParser(description="Module 03: Chunking exercise")
    parser.add_argument("--no-pause", action="store_true", help="Run without pauses")
    parser.add_argument("--data-dir", default=settings.data_dir, help="Data directory")
    args = parser.parse_args()

    if args.no_pause:
        os.environ["NO_PAUSE"] = "1"

    section("Module 03: Text Splitting & Chunking", "puzzle")
    info("Documents are too large to embed directly — we need to split them.")
    info("This module explores three splitting strategies.")

    # Load a document to work with
    documents = await load_directory(args.data_dir)
    if not documents:
        warning("No documents found. Run `python scripts/setup.py` first.")
        return

    # Pick a medium-sized document for demonstrations
    doc = sorted(documents, key=lambda d: len(d.page_content))[len(documents) // 2]
    result("Working with", doc.metadata.get("filename", "unknown"))
    result("Document size", f"{len(doc.page_content)} characters")

    wait_for_user()

    # ── Step 1: Character splitting ──
    step(1, "Fixed-size character splitting")
    info(f"Chunk size: {settings.chunk_size}, Overlap: {settings.chunk_overlap}")
    info("This splits text every N characters, regardless of word boundaries.")

    char_chunks = split_by_characters(
        doc, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    result("Chunks produced", len(char_chunks))
    for c in char_chunks[:3]:
        info(f'Chunk {c.chunk_index}: {len(c.content)} chars — "{c.content[:80]}..."')

    if len(char_chunks) > 1:
        # Show the boundary between chunk 0 and chunk 1
        end_of_0 = char_chunks[0].content[-30:]
        start_of_1 = char_chunks[1].content[:30]
        info(f'Boundary: ..."{end_of_0}" | "{start_of_1}"...')

    wait_for_user()

    # ── Step 2: Recursive splitting ──
    step(2, "Recursive character splitting")
    info("Uses separator hierarchy: paragraphs → lines → sentences → words → chars")
    info("This preserves semantic boundaries whenever possible.")

    recursive_chunks = split_recursively(
        doc, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    result("Chunks produced", len(recursive_chunks))
    for c in recursive_chunks[:3]:
        info(f'Chunk {c.chunk_index}: {len(c.content)} chars — "{c.content[:80]}..."')

    # Compare with character splitting
    char_cut = 0
    for cc in char_chunks:
        # Check if chunk ends mid-word (last char is a letter and next chunk starts with a letter)
        if cc.content and cc.content[-1].isalpha():
            char_cut += 1
    recursive_cut = 0
    for rc in recursive_chunks:
        if rc.content and rc.content[-1].isalpha():
            recursive_cut += 1

    info(f"Character splitter: {char_cut}/{len(char_chunks)} chunks end mid-word")
    info(
        f"Recursive splitter: {recursive_cut}/{len(recursive_chunks)} chunks end mid-word"
    )

    if recursive_cut < char_cut:
        success("Recursive splitter produces fewer mid-word cuts!")
    else:
        info("Similar results for this document size.")

    wait_for_user()

    # ── Step 3: Token-aware splitting ──
    step(3, "Token-aware splitting")
    chunk_tokens = 128
    info(f"Target: {chunk_tokens} tokens per chunk (~{chunk_tokens * 4} characters)")
    info("Uses word-level splitting with token count estimation.")

    token_chunks = split_by_tokens(
        doc, chunk_tokens=chunk_tokens, chunk_overlap_tokens=16
    )

    result("Chunks produced", len(token_chunks))
    rows = []
    for c in token_chunks[:5]:
        rows.append(
            [
                str(c.chunk_index),
                str(len(c.content)),
                str(c.metadata.get("approx_tokens", "?")),
            ]
        )
    table(
        headers=["Chunk", "Chars", "Est. Tokens"],
        rows=rows,
        title="Token-Aware Chunks",
    )

    result("Chars/token estimate", "~4.0 (English average)")
    info("Actual tokens depend on the model's tokenizer (BPE, WordPiece, etc.)")

    wait_for_user()

    # ── Step 4: Overlap comparison ──
    step(4, "Chunk overlap comparison")
    info("Overlap prevents key facts from being split across chunk boundaries.")
    info("Let's compare 0, 50, and 100 character overlap on the same document.")

    rows = []
    for overlap in [0, 50, 100]:
        chunks = split_by_characters(doc, chunk_size=500, chunk_overlap=overlap)
        rows.append([str(overlap), str(len(chunks)), f"{len(chunks)} chunks"])

    table(headers=["Overlap", "Chunks", "Notes"], rows=rows)
    info("More overlap = more chunks = more embeddings to store.")
    info("But overlap also = better chance of capturing split facts.")

    wait_for_user()

    # ── Step 5: Strategy comparison ──
    step(5, "Strategy comparison")
    info("Let's compare all three strategies on the same document.")

    strategies = [
        ("Character", char_chunks),
        ("Recursive", recursive_chunks),
        ("Token", token_chunks),
    ]

    rows = []
    for name, chunks in strategies:
        avg_len = sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0
        min_len = min(len(c.content) for c in chunks) if chunks else 0
        max_len = max(len(c.content) for c in chunks) if chunks else 0
        rows.append([name, str(len(chunks)), str(avg_len), f"{min_len}-{max_len}"])

    table(
        headers=["Strategy", "Chunks", "Avg Size", "Size Range"],
        rows=rows,
        title="Strategy Comparison",
    )

    divider()
    info("Character splitting is fastest but cuts mid-word.")
    info("Recursive splitting preserves semantic boundaries.")
    info("Token splitting is model-aware but approximate without a real tokenizer.")

    section("Exercise Complete!", "check")
    success("You've completed the Chunking exercise.")
    info("Next: Module 04 — Embeddings")
    info("Run: python -m rag_playground.m04_embeddings.exercise")


if __name__ == "__main__":
    asyncio.run(main())
