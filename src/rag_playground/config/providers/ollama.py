"""Ollama model provider implementation.

Communicates with Ollama's REST API using httpx (async HTTP client).
No ollama-python SDK — we talk directly to the HTTP endpoints.

Key Learning Points:
  - Ollama's API is simple REST — /api/chat, /api/embed, /api/tags
  - Streaming uses NDJSON (newline-delimited JSON) — one JSON object per line
  - httpx is the modern async Python HTTP client (like requests, but async)
  - Proper error handling for connection failures, timeouts, non-200 responses
"""

import json
import logging
from collections.abc import AsyncIterator

import httpx

from .base import ChatMessage, ChatOptions, EmbedResult, ModelProvider

logger = logging.getLogger(__name__)


class OllamaProvider(ModelProvider):
    """Ollama REST API provider — local LLM + embeddings via HTTP."""

    def __init__(self, host: str = "http://localhost:11434", timeout: float = 120.0):
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "ollama"

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-create the httpx client (shared connection pool)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._host,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Chat ---

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> str:
        """Send a non-streaming chat completion request."""
        opts = options or ChatOptions()
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": opts.temperature,
            },
        }
        if opts.max_tokens is not None:
            payload["options"]["num_predict"] = opts.max_tokens
        if opts.top_p is not None:
            payload["options"]["top_p"] = opts.top_p

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion. Yields each token as it arrives."""
        opts = options or ChatOptions()
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": opts.temperature,
            },
        }
        if opts.max_tokens is not None:
            payload["options"]["num_predict"] = opts.max_tokens
        if opts.top_p is not None:
            payload["options"]["top_p"] = opts.top_p

        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("done", False):
                        break
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    # --- Embeddings ---

    async def embed(self, model: str, texts: list[str]) -> EmbedResult:
        """Generate embeddings for a batch of texts.

        Ollama's /api/embed accepts a list of strings and returns
        a list of embedding vectors (one per input text).
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "input": texts,
        }

        response = await client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()

        vectors = data["embeddings"]
        return EmbedResult(
            vectors=vectors,
            model=model,
            dimensions=len(vectors[0]) if vectors else 0,
        )

    # --- Utility ---

    async def list_models(self) -> list[str]:
        """Return list of available model names from Ollama."""
        client = await self._get_client()

        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()

        return [m["name"] for m in data.get("models", [])]

    async def health_check(self) -> bool:
        """Check if Ollama is reachable by hitting the root endpoint."""
        try:
            client = await self._get_client()
            response = await client.get("/")
            return response.status_code == 200
        except Exception:
            return False
