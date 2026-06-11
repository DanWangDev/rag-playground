"""Model registry — resolves provider + model from environment.

This is the central configuration point. All modules call:
  - resolve_chat_config() → (provider, model_name)
  - resolve_embed_config() → (provider, model_name, dimensions)

Key Learning Points:
  - lru_cache means config is parsed once and reused — not on every call
  - Modules never import provider classes directly — they use this registry
  - Adding a new provider requires ONE change to this file (match case branch)
  - The registry pattern decouples modules from provider implementations
"""

from functools import lru_cache

from .env import settings
from .providers.base import ModelProvider
from .providers.ollama import OllamaProvider


@lru_cache(maxsize=1)
def resolve_chat_config() -> tuple[ModelProvider, str]:
    """Returns (provider_instance, model_name) for chat completions.

    Reads CHAT_PROVIDER and CHAT_MODEL from environment.
    Cached — same instance returned on every call.
    """
    match settings.chat_provider:
        case "ollama":
            provider = OllamaProvider(host=settings.ollama_host)
        case _:
            raise ValueError(
                f"Unknown chat provider: {settings.chat_provider!r}. Supported: ollama"
            )
    return provider, settings.chat_model


@lru_cache(maxsize=1)
def resolve_embed_config() -> tuple[ModelProvider, str, int]:
    """Returns (provider_instance, model_name, dimensions) for embeddings.

    Reads EMBED_PROVIDER, EMBED_MODEL, and EMBED_DIMENSIONS from environment.
    Cached — same instance returned on every call.
    """
    match settings.embed_provider:
        case "ollama":
            provider = OllamaProvider(host=settings.ollama_host)
        case _:
            raise ValueError(
                f"Unknown embed provider: {settings.embed_provider!r}. "
                f"Supported: ollama"
            )
    return provider, settings.embed_model, settings.embed_dimensions
