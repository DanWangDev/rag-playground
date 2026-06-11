"""Interactive exercise: Data Loading.

Usage:
    python -m rag_playground.m02_data_loading.exercise
    python -m rag_playground.m02_data_loading.exercise --no-pause
    python -m rag_playground.m02_data_loading.exercise --data-dir ./my-docs

Key Learning Points:
  - Every exercise follows the same pattern: section → step → result → pause
  - Users see the Document objects being created in real time
  - The --data-dir flag makes it work with user's own documents
"""

import argparse
import asyncio
import os

from rag_playground.config.env import settings
from rag_playground.m02_data_loading.document import load_markdown_file, load_text_file
from rag_playground.m02_data_loading.load_directory import load_directory
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
    parser = argparse.ArgumentParser(description="Module 02: Data Loading exercise")
    parser.add_argument("--no-pause", action="store_true", help="Run without pauses")
    parser.add_argument("--data-dir", default=settings.data_dir, help="Data directory")
    args = parser.parse_args()

    if args.no_pause:
        os.environ["NO_PAUSE"] = "1"

    section("Module 02: Data Loading", "file")
    info(
        "In this exercise, you'll load documents from files — "
        "the first step in any RAG pipeline."
    )
    info(
        "Documents are loaded into a standard format: "
        "page_content (the text) + metadata (source, filename, etc.)"
    )

    # ── Step 1: Load a plain text file ──
    step(1, "Loading a text file")
    info("First, let's load a simple text file.")

    # Find a sample file
    data_dir = args.data_dir
    text_files = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".txt"):
                text_files.append(os.path.join(root, f))

    if text_files:
        sample_path = text_files[0]
        info(f"Loading: {sample_path}")
        try:
            doc = await load_text_file(sample_path)
            result("Content preview", doc.page_content[:200] + "...")
            result("Metadata", "")
            for key, value in doc.metadata.items():
                print(f"      {key}: {value}")
            success("Text file loaded successfully!")
        except Exception as e:
            warning(f"Could not load text file: {e}")
    else:
        info(
            "No .txt files found in the data directory. "
            "We'll use the markdown files instead."
        )

    wait_for_user()

    # ── Step 2: Load a markdown file with frontmatter ──
    step(2, "Loading a markdown file (with frontmatter)")
    info("Markdown files often have YAML frontmatter at the top.")
    info("The loader extracts this as metadata automatically.")

    md_files = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))

    if md_files:
        sample_path = md_files[0]
        info(f"Loading: {sample_path}")
        doc = await load_markdown_file(sample_path)

        result("Title", doc.metadata.get("title", "(no title)"))
        result("Filename", doc.metadata.get("filename", "N/A"))
        result("Content preview", doc.page_content[:150] + "...")

        # Show all metadata
        info("All metadata fields:")
        for key, value in doc.metadata.items():
            print(f"      {key}: {value}")

        success("Markdown file loaded with metadata!")
    else:
        warning("No .md files found.")

    wait_for_user()

    # ── Step 3: Load an entire directory ──
    step(3, "Loading an entire directory")
    info(f"Now let's load ALL documents from: {data_dir}")
    info("This is how you'd index a knowledge base for RAG.")

    documents = await load_directory(data_dir)

    result("Total documents loaded", len(documents))

    # Show summary table
    rows = []
    for d in documents:
        rows.append(
            [
                d.metadata.get("filename", "unknown"),
                d.metadata.get("file_type", "?"),
                str(len(d.page_content)),
                d.metadata.get("top_level_dir", "-"),
            ]
        )
    table(
        headers=["Filename", "Type", "Chars", "Directory"],
        rows=rows,
        title="Loaded Documents",
    )

    success(f"Loaded {len(documents)} documents from directory!")

    wait_for_user()

    # ── Step 4: Filter documents by metadata ──
    step(4, "Filtering documents by metadata")
    info("Metadata lets you filter documents before chunking.")

    # Group by top-level directory
    dirs: dict[str, int] = {}
    for d in documents:
        dir_name = d.metadata.get("top_level_dir", "unknown")
        dirs[dir_name] = dirs.get(dir_name, 0) + 1

    info("Documents by directory:")
    for dir_name, count in sorted(dirs.items()):
        print(f"      {dir_name}: {count} documents")

    # Show FAQ docs
    faq_docs = [d for d in documents if d.metadata.get("top_level_dir") == "faq"]
    info(f"FAQ documents ({len(faq_docs)}):")
    for d in faq_docs:
        print(f"      📄 {d.metadata.get('filename')}")

    success("Metadata filtering works!")

    wait_for_user()

    # ── Step 5: Prepare for chunking ──
    step(5, "Preparing for Module 03 (Chunking)")
    info("Before we can chunk documents, we need to know their sizes.")
    info("This helps us choose the right chunk size in the next module.")

    total_chars = sum(len(d.page_content) for d in documents)
    avg_chars = total_chars // len(documents) if documents else 0
    smallest = min(len(d.page_content) for d in documents) if documents else 0
    largest = max(len(d.page_content) for d in documents) if documents else 0

    result("Total characters", total_chars)
    result("Average per document", avg_chars)
    result("Smallest document", smallest)
    result("Largest document", largest)

    divider()
    info(
        f"With a chunk size of {settings.chunk_size} characters and "
        f"{settings.chunk_overlap} overlap, you'd get roughly "
        f"{total_chars // (settings.chunk_size - settings.chunk_overlap)} chunks."
    )
    info("We'll explore chunking strategies in the next module!")

    section("Exercise Complete!", "check")
    success("You've completed the Data Loading exercise.")
    info("Next: Module 03 — Text Splitting & Chunking")
    info("Run: python -m rag_playground.m03_chunking.exercise")


if __name__ == "__main__":
    asyncio.run(main())
