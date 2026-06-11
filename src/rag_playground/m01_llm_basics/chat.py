"""Chat completion and streaming via the model provider.

Key Learning Points:
  - chat_completion returns the full response at once
  - chat_stream yields tokens as they arrive (real-time UX)
  - Both use the same provider abstraction — works with any backend
  - ChatOptions control temperature, max_tokens, top_p
"""

from collections.abc import AsyncIterator

from rag_playground.config.models import resolve_chat_config
from rag_playground.config.providers.base import ChatMessage, ChatOptions


async def chat_completion(
    messages: list[ChatMessage],
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> str:
    """Send a chat completion request. Returns the full response.

    Args:
        messages: List of chat messages (system, user, assistant).
        temperature: 0.0-2.0. Lower = more deterministic.
        max_tokens: Maximum tokens in the response (None = model default).

    Returns:
        The model's response text.
    """
    provider, model = resolve_chat_config()
    options = ChatOptions(temperature=temperature, max_tokens=max_tokens)
    return await provider.chat(model, messages, options)


async def chat_stream(
    messages: list[ChatMessage],
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion. Yields tokens as they arrive.

    Args:
        messages: List of chat messages.
        temperature: 0.0-2.0.
        max_tokens: Maximum tokens in the response.

    Yields:
        Text tokens as they arrive from the model.
    """
    provider, model = resolve_chat_config()
    options = ChatOptions(temperature=temperature, max_tokens=max_tokens)
    async for token in provider.chat_stream(model, messages, options):
        yield token
