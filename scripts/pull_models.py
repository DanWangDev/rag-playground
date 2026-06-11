"""Pull required Ollama models.

Usage:
    python scripts/pull_models.py
    python scripts/pull_models.py --chat-model qwen2.5:14b

Key Learning Points:
  - Ollama's /api/pull endpoint streams download progress as NDJSON
  - The script polls for completion, showing progress along the way
  - Models are pulled into Ollama's cache — not into this project directory
"""

import argparse
import asyncio
import sys

import httpx

from rag_playground.config.env import settings


async def pull_model(host: str, model: str) -> bool:
    """Pull a single model from Ollama. Returns True on success."""
    print(f"⬇️  Pulling {model}... (this may take a few minutes)")

    try:
        async with httpx.AsyncClient(base_url=host, timeout=600.0) as client:
            async with client.stream("POST", "/api/pull", json={"name": model}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json

                        chunk = json.loads(line)
                        status = chunk.get("status", "")
                        if status:
                            # Show progress for download status lines
                            if "downloading" in status:
                                completed = chunk.get("completed", 0)
                                total = chunk.get("total", 0)
                                if total:
                                    pct = completed / total * 100
                                    print(f"   {status} {pct:.0f}%", end="\r")
                                else:
                                    print(f"   {status}", end="\r")
                            else:
                                print(f"   {status}")
                    except (json.JSONDecodeError, KeyError):
                        continue

        print(f"✅ {model} pulled successfully")
        return True

    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to pull {model}: HTTP {e.response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Failed to pull {model}: {e}")
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="Pull required Ollama models")
    parser.add_argument("--chat-model", default=settings.chat_model, help="Chat model to pull")
    parser.add_argument("--embed-model", default=settings.embed_model, help="Embedding model to pull")
    args = parser.parse_args()

    print("🚀 Pulling required Ollama models...")
    print(f"   Chat model: {args.chat_model}")
    print(f"   Embed model: {args.embed_model}")
    print()

    results = await asyncio.gather(
        pull_model(settings.ollama_host, args.chat_model),
        pull_model(settings.ollama_host, args.embed_model),
    )

    if all(results):
        print()
        print("✅ All models pulled successfully")
    else:
        print()
        print("⚠️  Some models failed to pull. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
