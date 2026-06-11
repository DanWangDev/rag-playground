"""RAG answer generation — retrieve + augment + generate.

Key Learning Points:
  - generate_answer combines retrieval (Module 06) + generation (Module 01)
  - prompt_templates format retrieved context for the LLM
  - The system prompt is critical: constrains LLM to use only provided context
  - generate_streaming_answer provides token-by-token output
  - Optional provider params enable testing without a real Ollama server
"""

import time
from collections.abc import AsyncIterator

from rag_playground.config.providers.base import ChatMessage, ModelProvider
from rag_playground.m01_llm_basics.chat import chat_completion, chat_stream
from rag_playground.m06_retrieval.basic_retrieval import retrieve
from rag_playground.m05_vector_store.in_memory_store import InMemoryVectorStore
from rag_playground.m07_rag_pipeline.prompt_templates import (
    QA_PROMPT,
    CITATION_PROMPT,
    format_context,
)
from rag_playground.shared.types import RAGResponse


async def generate_answer(
    query: str,
    store: InMemoryVectorStore,
    top_k: int = 3,
    system_prompt: str | None = None,
    embed_provider: ModelProvider | None = None,
    embed_model: str | None = None,
    chat_provider: ModelProvider | None = None,
    chat_model: str | None = None,
) -> RAGResponse:
    """Full RAG pipeline: retrieve context, augment prompt, generate answer.

    Args:
        query: The user's question.
        store: Vector store with indexed documents.
        top_k: Number of chunks to retrieve.
        system_prompt: Custom system prompt (uses QA_PROMPT by default).
        embed_provider: Optional embed provider (for testing).
        embed_model: Optional embed model name.
        chat_provider: Optional chat provider (for testing).
        chat_model: Optional chat model name.

    Returns:
        RAGResponse with answer, sources, and token estimates.
    """
    start = time.perf_counter()

    # Step 1: Retrieve relevant chunks
    retrieval_result = await retrieve(
        query,
        store,
        top_k=top_k,
        embed_provider=embed_provider,
        embed_model=embed_model,
    )

    # Step 2: Format retrieved context
    context = format_context(retrieval_result.results)

    # Step 3: Build the prompt
    sys_prompt = system_prompt or QA_PROMPT
    messages = [
        ChatMessage(role="system", content=sys_prompt),
        ChatMessage(
            role="user",
            content=CITATION_PROMPT.format(context=context, question=query),
        ),
    ]

    # Step 4: Generate answer
    if chat_provider is not None and chat_model is not None:
        answer = await chat_provider.chat(chat_model, messages)
    else:
        answer = await chat_completion(messages, temperature=0.1)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return RAGResponse(
        answer=answer,
        sources=retrieval_result.results,
        prompt_tokens=len(context) // 4,  # Approximate
        completion_tokens=len(answer) // 4,
        elapsed_ms=elapsed_ms,
    )


async def generate_streaming_answer(
    query: str,
    store: InMemoryVectorStore,
    top_k: int = 3,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Streaming RAG: retrieve, then stream the generated answer token by token.

    Args:
        query: The user's question.
        store: Vector store with indexed documents.
        top_k: Number of chunks to retrieve.
        system_prompt: Custom system prompt.

    Yields:
        Answer tokens as they arrive from the LLM.
    """
    # Retrieve context
    retrieval_result = await retrieve(query, store, top_k=top_k)
    context = format_context(retrieval_result.results)

    # Build prompt
    sys_prompt = system_prompt or QA_PROMPT
    messages = [
        ChatMessage(role="system", content=sys_prompt),
        ChatMessage(
            role="user",
            content=CITATION_PROMPT.format(context=context, question=query),
        ),
    ]

    # Stream the answer
    async for token in chat_stream(messages, temperature=0.1):
        yield token
