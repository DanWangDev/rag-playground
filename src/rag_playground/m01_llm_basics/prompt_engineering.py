"""Prompt engineering utilities.

Key Learning Points:
  - System prompts set the AI's behavior/persona
  - Few-shot prompting provides examples to guide output format
  - Chain-of-thought improves reasoning by asking for step-by-step thinking
  - These patterns are essential for RAG (constraining answers to context)
"""

from rag_playground.config.providers.base import ChatMessage


def with_system_prompt(user_message: str, system_prompt: str) -> list[ChatMessage]:
    """Create messages with a system prompt.

    System prompts set the AI's behavior and constraints.
    In RAG, this is where you tell the model to ONLY use provided context.

    Args:
        user_message: The user's question or instruction.
        system_prompt: The system-level instruction.

    Returns:
        List of ChatMessage objects ready for chat_completion().
    """
    return [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    ]


def with_few_shot(
    user_message: str,
    examples: list[tuple[str, str]],
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    """Create messages with few-shot examples.

    Few-shot prompting shows the model examples of the desired behavior.
    The model learns the pattern from the examples and applies it to the new input.

    Args:
        user_message: The actual query.
        examples: List of (input, output) pairs to demonstrate the pattern.
        system_prompt: Optional system-level instruction.

    Returns:
        List of ChatMessage objects ready for chat_completion().
    """
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))

    for user_example, assistant_example in examples:
        messages.append(ChatMessage(role="user", content=user_example))
        messages.append(ChatMessage(role="assistant", content=assistant_example))

    messages.append(ChatMessage(role="user", content=user_message))
    return messages


def with_chain_of_thought(
    user_message: str,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    """Create messages that prompt chain-of-thought reasoning.

    Adding "Let's think step by step" significantly improves
    reasoning accuracy on complex questions.

    Args:
        user_message: The question to reason about.
        system_prompt: Optional system instruction.

    Returns:
        List of ChatMessage objects.
    """
    cot_prompt = f"{user_message}\n\nLet's think about this step by step."
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=cot_prompt))
    return messages
