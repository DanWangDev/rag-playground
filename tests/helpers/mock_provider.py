"""Mock ModelProvider for unit testing.

Provides deterministic responses so tests run fast and offline.
No Ollama server needed — tests verify logic, not HTTP calls.

Key Learning Points:
  - Mocks implement the same ABC as real providers (Liskov substitution)
  - Deterministic responses make tests reliable and repeatable
  - Mock embeddings use simple hash-based vectors — fast but not semantically meaningful
"""

from collections.abc import AsyncIterator

from rag_playground.config.providers.base import (
    ChatMessage,
    ChatOptions,
    EmbedResult,
    ModelProvider,
)


class MockProvider(ModelProvider):
    """Fake provider that returns deterministic responses.

    Embeddings are derived from string length using a simple hash —
    not semantically meaningful, but deterministic and fast.
    """

    @property
    def name(self) -> str:
        return "mock"

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> str:
        """Return a deterministic response based on the last message."""
        if not messages:
            return "No messages provided."

        last = messages[-1].content.lower()
        # Deterministic responses for known test queries
        if "machine learning" in last:
            return "Machine learning is a field of AI that learns from data."
        if "climate" in last:
            return "Climate change is caused by greenhouse gas emissions."
        if "world war" in last:
            return "World War II lasted from 1939 to 1945."
        if "renaissance" in last:
            return "The Renaissance was a period of cultural rebirth in Europe."
        return f"Mock response to: {messages[-1].content[:50]}..."

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[str]:
        """Stream a deterministic response word by word."""
        text = await self.chat(model, messages, options)
        for word in text.split():
            yield word + " "

    async def embed(self, model: str, texts: list[str]) -> EmbedResult:
        """Generate deterministic mock embeddings.

        Each text gets a 4-dimensional vector derived from its hash.
        This is fast and deterministic but NOT semantically meaningful.
        Real embeddings (768d from nomic-embed-text) behave similarly
        at the API level — just with real semantic capture.
        """
        vectors: list[list[float]] = []
        for text in texts:
            # Simple deterministic vector: length + character-based values
            h = abs(hash(text)) % 1000
            vec = [
                (h % 100) / 100.0,
                ((h // 10) % 100) / 100.0,
                ((h // 100) % 100) / 100.0,
                (len(text) % 100) / 100.0,
            ]
            # Normalize to unit length (like real embeddings)
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        return EmbedResult(
            vectors=vectors,
            model=model,
            dimensions=len(vectors[0]) if vectors else 0,
        )

    async def list_models(self) -> list[str]:
        return ["mock-model", "mock-embed-model"]

    async def health_check(self) -> bool:
        return True
