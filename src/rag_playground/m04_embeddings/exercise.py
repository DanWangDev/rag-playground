"""Interactive exercise: Embeddings.

Usage:
    python -m rag_playground.m04_embeddings.exercise
    python -m rag_playground.m04_embeddings.exercise --no-pause
"""

import argparse
import asyncio
import os

from rag_playground.config.env import settings
from rag_playground.config.models import resolve_embed_config
from rag_playground.m04_embeddings.embed import embed_documents, embed_query
from rag_playground.m04_embeddings.similarity import (
    cosine_similarity,
    find_most_similar,
    similarity_matrix,
)
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
    parser = argparse.ArgumentParser(description="Module 04: Embeddings exercise")
    parser.add_argument("--no-pause", action="store_true", help="Run without pauses")
    args = parser.parse_args()

    if args.no_pause:
        os.environ["NO_PAUSE"] = "1"

    section("Module 04: Embeddings", "brain")
    provider, model, dims = resolve_embed_config()
    info(f"Embedding model: {model} ({dims} dimensions)")
    info("Embeddings convert text into vectors — numbers that capture meaning.")

    # Check if Ollama is reachable
    healthy = await provider.health_check()
    if not healthy:
        warning(f"Cannot reach Ollama at {settings.ollama_host}")
        info("Start Ollama with: ollama serve")
        info("Then pull models: python scripts/pull_models.py")
        info("Skipping embedding generation — showing structure only.")
        return

    wait_for_user()

    # ── Step 1: Generate embeddings for sentences ──
    step(1, "Generating embeddings")
    sentences = [
        "Machine learning is a field of artificial intelligence.",
        "AI systems can learn from data without being explicitly programmed.",
        "Climate change refers to long-term shifts in global temperatures.",
        "The Earth's climate has been warming due to greenhouse gas emissions.",
        "The Renaissance was a period of European cultural and artistic rebirth.",
        "Renaissance artists like Leonardo da Vinci created masterpieces.",
    ]

    info(f"Embedding {len(sentences)} sentences using {model}...")
    vectors = await embed_documents(sentences)

    result("Sentences embedded", len(vectors))
    result("Vector dimensions", len(vectors[0]))
    result("Sample vector (first 5 values)", str(vectors[0][:5]) + "...")

    info("Each sentence is now a point in 768-dimensional space.")
    info("Similar sentences should be close together!")

    wait_for_user()

    # ── Step 2: Compute pairwise similarity ──
    step(2, "Computing cosine similarity")
    info("Cosine similarity measures how close two vectors are.")
    info("1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite.")

    # ML sentence pair
    sim_ml = cosine_similarity(vectors[0], vectors[1])
    # Climate sentence pair
    sim_climate = cosine_similarity(vectors[2], vectors[3])
    # Cross-topic pair
    sim_cross = cosine_similarity(vectors[0], vectors[4])

    rows = [
        ["ML sentences (same topic)", f"{sim_ml:.4f}"],
        ["Climate sentences (same topic)", f"{sim_climate:.4f}"],
        ["ML vs Renaissance (different)", f"{sim_cross:.4f}"],
    ]
    table(headers=["Pair", "Similarity"], rows=rows, title="Cosine Similarity")

    if sim_ml > sim_cross and sim_climate > sim_cross:
        success("Same-topic pairs are more similar — embeddings work!")
    else:
        info("Unexpected results — embedding model may differ.")

    wait_for_user()

    # ── Step 3: Find nearest neighbors ──
    step(3, "Finding nearest neighbors")
    info("Given a query, we find the most similar sentences in our collection.")

    query = "How does artificial intelligence learn?"
    info(f'Query: "{query}"')
    query_vec = await embed_query(query)

    top = find_most_similar(query_vec, vectors, top_k=3)

    info("Top 3 matches:")
    for rank, (idx, score) in enumerate(top, 1):
        info(f'  #{rank} (score: {score:.4f}): "{sentences[idx]}"')

    success("Nearest neighbor search complete!")

    wait_for_user()

    # ── Step 4: Similarity matrix ──
    step(4, "Similarity matrix")

    labels = ["ML-1", "ML-2", "Climate-1", "Climate-2", "Art-1", "Art-2"]
    matrix = similarity_matrix(vectors)

    info("Pairwise similarity matrix (darker = more similar):")
    # Print a simple text-based matrix
    print()
    header = "         " + " ".join(f"{label:>8}" for label in labels)
    print(header)
    print("         " + "-" * (len(labels) * 9))
    for i, label in enumerate(labels):
        row = f"{label:>7} |"
        for j in range(len(labels)):
            val = matrix[i][j]
            # Visual indicator
            if val > 0.8:
                bar = "██"
            elif val > 0.6:
                bar = "▓▓"
            elif val > 0.4:
                bar = "░░"
            else:
                bar = "  "
            row += f" {val:.2f}{bar}"
        print(row)
    print()

    info("Notice how same-topic blocks (ML/Climate/Art) form dark clusters.")
    info("This clustering is what makes retrieval work!")

    wait_for_user()

    # ── Step 5: Paraphrase detection ──
    step(5, "Paraphrase detection")
    info("Embeddings can detect when different words mean the same thing.")

    original = "The quick brown fox jumps over the lazy dog."
    paraphrase = "A fast auburn fox leaps above a sleepy canine."
    unrelated = "The stock market crashed on Tuesday afternoon."

    texts = [original, paraphrase, unrelated]
    vecs = await embed_documents(texts)

    sim_para = cosine_similarity(vecs[0], vecs[1])
    sim_unrel = cosine_similarity(vecs[0], vecs[2])

    rows = [
        ["Original vs Paraphrase", f"{sim_para:.4f}", "✅ Same meaning"],
        ["Original vs Unrelated", f"{sim_unrel:.4f}", "❌ Different meaning"],
    ]
    table(headers=["Pair", "Similarity", "Verdict"], rows=rows)

    divider()
    info(
        "Embeddings capture meaning, not just words — this is the foundation of semantic search."
    )

    section("Exercise Complete!", "check")
    success("You've completed the Embeddings exercise.")
    info("Next: Module 05 — Vector Store")
    info("Run: python -m rag_playground.m05_vector_store.exercise")


if __name__ == "__main__":
    asyncio.run(main())
