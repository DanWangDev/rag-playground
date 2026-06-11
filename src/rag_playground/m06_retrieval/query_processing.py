"""Query preprocessing strategies.

Key Learning Points:
  - expand_query: LLM generates related terms for broader search
  - decompose_query: LLM breaks complex questions into simpler sub-questions
  - hyde_query: Generate a hypothetical answer, embed that as the query
"""

from rag_playground.config.providers.base import ChatMessage


async def expand_query(query: str, chat_fn) -> str:
    """Expand a query with related terms and synonyms using an LLM.

    Example: "ML" → "machine learning, artificial intelligence, deep learning"

    Args:
        query: The original query.
        chat_fn: Async function that takes messages and returns a string.

    Returns:
        An expanded version of the query with additional search terms.
    """
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are a query expansion tool. Given a search query, generate "
                "a space-separated list of synonyms, related terms, and alternative "
                "phrasings that would help find relevant documents. "
                "Output ONLY the expanded terms, no explanation."
            ),
        ),
        ChatMessage(role="user", content=f"Expand: {query}"),
    ]
    expansion = await chat_fn(messages)
    return f"{query} {expansion.strip()}"


async def decompose_query(query: str, chat_fn) -> list[str]:
    """Decompose a complex query into simpler sub-questions.

    Example: "Compare ML and quantum computing" →
      ["What is machine learning?", "What is quantum computing?"]

    Args:
        query: The complex query.
        chat_fn: Async chat function.

    Returns:
        List of simpler sub-queries.
    """
    messages = [
        ChatMessage(
            role="system",
            content=(
                "Break down complex questions into simpler, self-contained sub-questions. "
                "Output one question per line. No numbering, no explanation."
            ),
        ),
        ChatMessage(role="user", content=f"Decompose: {query}"),
    ]
    response = await chat_fn(messages)
    return [
        line.strip("- ").strip()
        for line in response.strip().split("\n")
        if line.strip()
    ]


async def hyde_query(query: str, chat_fn, embed_fn) -> list[float]:
    """Hypothetical Document Embeddings (HyDE).

    Instead of embedding the query directly, we:
    1. Ask the LLM to generate a hypothetical answer to the query
    2. Embed the hypothetical answer
    3. Search for real documents similar to the hypothetical answer

    This works because a detailed answer is closer in embedding space
    to real relevant documents than a short query is.

    Args:
        query: The user's query.
        chat_fn: Async chat function for generating the hypothetical.
        embed_fn: Async embed function for vectorizing the hypothetical.

    Returns:
        The embedding vector of the hypothetical answer.
    """
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are a helpful assistant. Write a detailed, factual passage "
                "that answers the user's question. Write as if you are a knowledgeable "
                "expert, but do NOT say 'as an expert' or similar. Just provide the information."
            ),
        ),
        ChatMessage(role="user", content=query),
    ]
    hypothetical = await chat_fn(messages)
    return await embed_fn(hypothetical)
