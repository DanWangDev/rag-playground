"""Abstract ModelProvider base class.

Defines the contract that all model providers must implement.
Adding a new provider (e.g., OpenAI, vLLM) means implementing this ABC —
no changes needed in any module code.

Key Learning Points:
  - ABC (Abstract Base Class) enforces a contract at the language level
  - AsyncIterator enables memory-efficient streaming of tokens
  - embed_single() has a default implementation — subclasses only need embed()
  - Dataclasses keep message/options structures clean and typed
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatOptions:
    """Optional parameters for chat completion."""

    temperature: float = 0.1
    max_tokens: int | None = None
    top_p: float | None = None


@dataclass
class EmbedResult:
    """Result from an embedding request."""

    vectors: list[list[float]]
    model: str
    dimensions: int


class ModelProvider(ABC):
    """Abstract interface for LLM + embedding model providers.

    To add a new provider:
    1. Subclass this and implement all abstract methods
    2. Add a case branch in config/models.py
    3. That's it — all modules use the new provider automatically
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g., 'ollama', 'openai')."""
        ...

    # --- Chat ---

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> str:
        """Send a chat completion request. Returns the full response text."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion. Yields tokens as they arrive."""
        ...

    # --- Embeddings ---

    @abstractmethod
    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> EmbedResult:
        """Generate embeddings for a batch of texts."""
        ...

    async def embed_single(self, model: str, text: str) -> list[float]:
        """Generate embedding for a single text. Convenience wrapper."""
        result = await self.embed(model, [text])
        return result.vectors[0]

    # --- Utility ---

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return list of available model names from this provider."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and healthy."""
        ...
