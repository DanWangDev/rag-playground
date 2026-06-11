"""Module 01: LLM Basics — local chat, streaming, and prompt engineering."""

from .chat import chat_completion, chat_stream
from .prompt_engineering import with_chain_of_thought, with_few_shot, with_system_prompt

__all__ = [
    "chat_completion",
    "chat_stream",
    "with_system_prompt",
    "with_few_shot",
    "with_chain_of_thought",
]
