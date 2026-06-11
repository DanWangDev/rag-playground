"""RAG prompt templates.

Key Learning Points:
  - QA_PROMPT: constrains the LLM to use ONLY the provided context
  - CITATION_PROMPT: asks for source-attributed answers
  - SUMMARIZE_PROMPT: summarizes context without a specific question
  - Prompt engineering is critical for RAG quality — bad prompts = bad answers
"""

from rag_playground.shared.types import SearchResult

# Standard QA prompt: force the LLM to use context only
QA_PROMPT = """You are a helpful, accurate assistant. Answer the user's question using ONLY the provided context. Follow these rules:

1. If the context contains the answer, provide it accurately with citations.
2. If the context partially answers the question, provide what you can and note what's missing.
3. If the context does NOT contain enough information, say: "I don't have enough information to answer that question based on the provided documents."

Do NOT use your training data to answer. Only use the context below."""

# Citation prompt: formats context and asks for source-attributed answers
CITATION_PROMPT = """Context:
{context}

Question: {question}

Provide a clear, well-structured answer. Reference specific parts of the context where possible (e.g., "According to the first passage...").

Answer:"""

# Summarization prompt
SUMMARIZE_PROMPT = """You are a summarization assistant. Using ONLY the provided context below, write a concise summary covering the key points.

Context:
{context}

Summary:"""


def format_context(results: list[SearchResult]) -> str:
    """Format search results into a context block for the LLM prompt.

    Each chunk is labeled with its rank and score for traceability.

    Args:
        results: Ranked list of retrieval results.

    Returns:
        Formatted string with labeled passages.
    """
    if not results:
        return "(No relevant context found.)"

    parts: list[str] = []
    for r in results:
        parts.append(
            f"[Passage {r.rank} — relevance: {r.score:.2f}]\n{r.chunk.content}"
        )

    return "\n\n".join(parts)
