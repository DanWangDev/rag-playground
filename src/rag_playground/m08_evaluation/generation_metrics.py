"""Generation quality evaluation using LLM-as-judge.

Key Learning Points:
  - faithfulness: Is the answer grounded in the provided context?
  - relevance: Does the answer address the question?
  - LLM-as-judge uses the LLM itself to score quality
  - This is approximate but useful for comparing RAG configurations
"""

from rag_playground.config.providers.base import ChatMessage


async def faithfulness_score(
    answer: str,
    context: str,
    chat_fn,
) -> float:
    """Score how faithful the answer is to the provided context.

    Uses LLM-as-judge: asks the LLM to rate on a 1-5 scale
    whether the answer is fully supported by the context.

    Args:
        answer: The generated answer.
        context: The retrieved context used to generate the answer.
        chat_fn: Async chat function.

    Returns:
        Score from 0.0 (hallucinated) to 1.0 (fully faithful).
    """
    if not context.strip():
        return 0.0

    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are evaluating the faithfulness of an AI-generated answer. "
                "Rate whether the answer contains ONLY information present in the context. "
                "Score from 1-5:\n"
                "1 = Completely hallucinated, nothing from context\n"
                "3 = Some from context, some made up\n"
                "5 = Entirely grounded in the context\n"
                "Respond with ONLY the number (1-5)."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"Context:\n{context}\n\nAnswer:\n{answer}",
        ),
    ]
    response = await chat_fn(messages)
    try:
        score = float(response.strip()) / 5.0
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5


async def relevance_score(
    answer: str,
    question: str,
    chat_fn,
) -> float:
    """Score how relevant the answer is to the question.

    Uses LLM-as-judge to rate whether the answer addresses the question.

    Args:
        answer: The generated answer.
        question: The original user question.
        chat_fn: Async chat function.

    Returns:
        Score from 0.0 (irrelevant) to 1.0 (fully relevant).
    """
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are evaluating answer relevance. "
                "Rate whether the answer directly addresses the question. "
                "Score from 1-5:\n"
                "1 = Completely off-topic\n"
                "3 = Partially addresses the question\n"
                "5 = Directly and completely answers the question\n"
                "Respond with ONLY the number (1-5)."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"Question: {question}\n\nAnswer: {answer}",
        ),
    ]
    response = await chat_fn(messages)
    try:
        score = float(response.strip()) / 5.0
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5
