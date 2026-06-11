"""Verify Ollama is running and the required models are available.

Usage:
    python scripts/check_ollama.py
    python scripts/check_ollama.py --verbose

Key Learning Points:
  - Scripts are standalone Python files (no package prefix needed)
  - asyncio.run() is the standard way to enter async from sync code
  - Health checks should fail with clear, actionable error messages
"""

import argparse
import asyncio
import sys

from rag_playground.config.env import settings
from rag_playground.config.providers.ollama import OllamaProvider


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check Ollama connectivity and models")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    args = parser.parse_args()

    print("🔍 Checking Ollama connectivity...")
    provider = OllamaProvider(host=settings.ollama_host)

    # Health check
    healthy = await provider.health_check()
    if not healthy:
        print(f"❌ Cannot reach Ollama at {settings.ollama_host}")
        print("   Is Ollama running? Run `ollama serve` to start it.")
        sys.exit(1)

    print(f"✅ Ollama is running at {settings.ollama_host}")

    # List models
    try:
        models = await provider.list_models()
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"   Available models: {', '.join(models) if models else '(none)'}")

    # Check required models
    chat_model = settings.chat_model
    embed_model = settings.embed_model

    chat_available = any(m.startswith(chat_model) or chat_model in m for m in models)
    embed_available = any(m.startswith(embed_model) or embed_model in m for m in models)

    if chat_available:
        print(f"✅ Chat model '{chat_model}' is available")
    else:
        print(f"⚠️  Chat model '{chat_model}' is NOT pulled yet")
        print(f"   Run: ollama pull {chat_model}")

    if embed_available:
        print(f"✅ Embed model '{embed_model}' is available")
    else:
        print(f"⚠️  Embed model '{embed_model}' is NOT pulled yet")
        print(f"   Run: ollama pull {embed_model}")

    if not (chat_available and embed_available):
        print()
        print(
            "💡 Tip: Run `python scripts/pull_models.py` to pull all required models."
        )

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
