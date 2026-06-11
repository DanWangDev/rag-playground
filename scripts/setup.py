"""Full environment bootstrap.

Usage:
    python scripts/setup.py
    python scripts/setup.py --skip-pull   # Skip model pulling
    python scripts/setup.py --data-dir ./my-docs

Key Learning Points:
  - Setup scripts automate multi-step bootstrap processes
  - Each step is idempotent — safe to run multiple times
  - Failures at each step are reported clearly, not silently swallowed
"""

import argparse
import asyncio
import sys

from rag_playground.config.env import settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the RAG playground environment")
    parser.add_argument("--skip-pull", action="store_true", help="Skip pulling models")
    parser.add_argument("--data-dir", default=settings.data_dir, help="Data directory path")
    args = parser.parse_args()

    print("🚀 RAG Playground Setup")
    print("═" * 50)
    print()

    # Step 1: Check Ollama
    print("Step 1/3: Checking Ollama...")
    from rag_playground.config.providers.ollama import OllamaProvider

    provider = OllamaProvider(host=settings.ollama_host)
    healthy = await provider.health_check()
    if not healthy:
        print(f"❌ Ollama is not reachable at {settings.ollama_host}")
        print("   Start Ollama with: ollama serve")
        sys.exit(1)
    print(f"✅ Ollama running at {settings.ollama_host}")

    # Step 2: Pull models
    if not args.skip_pull:
        print()
        print("Step 2/3: Pulling models...")
        from scripts.pull_models import pull_model

        results = await asyncio.gather(
            pull_model(settings.ollama_host, settings.chat_model),
            pull_model(settings.ollama_host, settings.embed_model),
        )
        if not all(results):
            print("⚠️  Some models failed to pull. You can retry with:")
            print("   python scripts/pull_models.py")
    else:
        print()
        print("Step 2/3: Skipping model pull (--skip-pull)")

    # Step 3: Verify data directory
    print()
    print("Step 3/3: Checking data directory...")
    import os

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"⚠️  Data directory '{data_dir}' not found")
        print("   Creating the directory now...")
        os.makedirs(data_dir, exist_ok=True)

    doc_count = 0
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith((".md", ".txt")):
                doc_count += 1

    print(f"✅ Found {doc_count} documents in {data_dir}")

    await provider.close()

    print()
    print("═" * 50)
    print("✅ Setup complete! Run an exercise to get started:")
    print("   python -m rag_playground.m02_data_loading.exercise")


if __name__ == "__main__":
    asyncio.run(main())
