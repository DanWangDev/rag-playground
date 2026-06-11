"""Interactive exercise: Vector Store.

Usage:
    python -m rag_playground.m05_vector_store.exercise
    python -m rag_playground.m05_vector_store.exercise --no-pause
"""

import argparse
import asyncio
import os
import tempfile

from rag_playground.config.models import resolve_embed_config
from rag_playground.m04_embeddings.embed import embed_documents, embed_query
from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.m05_vector_store.persistence import load_from_file, save_to_file
from rag_playground.shared.logger import (
    info,
    result,
    section,
    step,
    success,
    warning,
)
from rag_playground.shared.prompt import wait_for_user
from rag_playground.shared.types import Chunk


async def main() -> None:
    parser = argparse.ArgumentParser(description="Module 05: Vector Store exercise")
    parser.add_argument("--no-pause", action="store_true", help="Run without pauses")
    args = parser.parse_args()

    if args.no_pause:
        os.environ["NO_PAUSE"] = "1"

    section("Module 05: Vector Store", "db")
    provider, model, dims = resolve_embed_config()
    info(f"Using embedding model: {model} ({dims}d)")

    healthy = await provider.health_check()
    if not healthy:
        warning("Ollama not reachable — demo uses mock data.")
        # Create mock embeddings (simplified for demo)
        mock_sentences = [
            "Machine learning is a field of AI.",
            "Climate change causes global warming.",
            "The Renaissance was a cultural movement.",
            "World War II ended in 1945.",
        ]
        # Use random-ish but deterministic "embeddings" for demo
        vectors = [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.1, 0.9],
        ]
        dims = 4
    else:
        mock_sentences = [
            "Machine learning is a field of artificial intelligence.",
            "Climate change refers to long-term shifts in global temperatures.",
            "The Renaissance was a period of European cultural rebirth.",
            "World War II was a global conflict from 1939 to 1945.",
        ]
        vectors = await embed_documents(mock_sentences)
        dims = len(vectors[0])

    wait_for_user()

    # ── Step 1: Create and populate the store ──
    step(1, "Creating an in-memory vector store")
    store = InMemoryVectorStore(embedding_model=model, dimensions=dims)

    for i, (vec, text) in enumerate(zip(vectors, mock_sentences)):
        chunk = Chunk(
            id=f"doc-{i}",
            content=text,
            document_id=f"source-{i}.md",
            metadata={"topic": ["ml", "climate", "art", "history"][i]},
            chunk_index=0,
        )
        store.add(vec, chunk, {"added_at": f"2024-01-{15 + i}"})

    result("Store size", store.size)
    result("Dimensions", store.dimensions)
    success("Vector store populated!")

    wait_for_user()

    # ── Step 2: Search ──
    step(2, "Searching the store")
    query = "Tell me about artificial intelligence"
    info(f'Query: "{query}"')

    query_vec = await embed_query(query) if healthy else [0.95, 0.05, 0.0, 0.0]
    results = store.search(query_vec, top_k=3)

    info("Top results:")
    for r in results:
        info(f'  #{r.rank} (score: {r.score:.4f}): "{r.chunk.content[:80]}..."')

    wait_for_user()

    # ── Step 3: Threshold filtering ──
    step(3, "Similarity threshold filtering")
    info("min_score filters out low-relevance results.")

    results_threshold = store.search(query_vec, top_k=10, min_score=0.5)
    result("Results above 0.5 threshold", len(results_threshold))

    results_strict = store.search(query_vec, top_k=10, min_score=0.9)
    result("Results above 0.9 threshold", len(results_strict))

    wait_for_user()

    # ── Step 4: Metadata filtering ──
    step(4, "Metadata filtering")
    info("Filter results to only include specific topics.")

    for topic in ["ml", "climate", "history", "art"]:
        results_filtered = store.search(
            query_vec, top_k=3, metadata_filter={"topic": topic}
        )
        count = len(results_filtered)
        score_str = f"{results_filtered[0].score:.4f}" if results_filtered else "N/A"
        info(f"  topic={topic}: {count} result(s), top score={score_str}")

    wait_for_user()

    # ── Step 5: Persistence ──
    step(5, "Saving and loading")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        save_to_file(store, tmp_path)
        file_size = os.path.getsize(tmp_path)
        result("Saved to", tmp_path)
        result("File size", f"{file_size} bytes")

        loaded = load_from_file(tmp_path)
        result("Loaded store size", loaded.size)
        result("Loaded dimensions", loaded.dimensions)
        assert loaded.size == store.size
        success("Save and load round-trip successful!")
    finally:
        os.unlink(tmp_path)

    section("Exercise Complete!", "check")
    success("Vector Store exercise complete.")
    info("Next: Module 06 — Retrieval Strategies")


if __name__ == "__main__":
    asyncio.run(main())
