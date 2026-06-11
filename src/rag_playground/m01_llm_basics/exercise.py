"""Interactive exercise: LLM Basics.

Usage:
    python -m rag_playground.m01_llm_basics.exercise
"""

import argparse
import asyncio
import os

from rag_playground.config.env import settings
from rag_playground.config.models import resolve_chat_config
from rag_playground.config.providers.base import ChatMessage
from rag_playground.m01_llm_basics.chat import chat_completion, chat_stream
from rag_playground.m01_llm_basics.prompt_engineering import (
    with_chain_of_thought,
    with_system_prompt,
)
from rag_playground.shared.logger import (
    info,
    result,
    section,
    step,
    success,
    warning,
)
from rag_playground.shared.prompt import wait_for_user


async def main() -> None:
    parser = argparse.ArgumentParser(description="Module 01: LLM Basics exercise")
    parser.add_argument("--no-pause", action="store_true", help="Run without pauses")
    args = parser.parse_args()

    if args.no_pause:
        os.environ["NO_PAUSE"] = "1"

    section("Module 01: LLM Basics", "robot")
    provider, model = resolve_chat_config()
    info(f"Using model: {model}")
    info("This module teaches you how to work with a local LLM via Ollama.")

    healthy = await provider.health_check()
    if not healthy:
        warning(f"Cannot reach Ollama at {settings.ollama_host}")
        info("Start with: ollama serve")
        return

    wait_for_user()

    # Step 1: Basic chat
    step(1, "Basic chat completion")
    messages = [
        ChatMessage(role="user", content="What is machine learning in one sentence?")
    ]
    info("Sending: What is machine learning in one sentence?")
    response = await chat_completion(messages, temperature=0.1)
    result("Response", response.strip())
    success("LLM responded!")

    wait_for_user()

    # Step 2: Temperature comparison
    step(2, "Temperature comparison")
    prompt = "Write a short creative slogan for a tech company."
    info(f'Same prompt, different temperatures: "{prompt}"')

    for temp in [0.0, 1.0]:
        msgs = [ChatMessage(role="user", content=prompt)]
        resp = await chat_completion(msgs, temperature=temp)
        info(f'  temp={temp}: "{resp.strip()[:100]}..."')

    info("temp=0.0 is more focused; temp=1.0 is more creative.")

    wait_for_user()

    # Step 3: Streaming
    step(3, "Streaming chat")
    messages = [ChatMessage(role="user", content="Count from 1 to 5, one per line.")]
    info("Streaming tokens as they arrive:")
    async for token in chat_stream(messages):
        print(token, end="", flush=True)
    print()
    success("Streaming works — tokens appear in real-time!")

    wait_for_user()

    # Step 4: System prompts
    step(4, "System prompts")
    question = "What is the capital of France?"
    msgs = with_system_prompt(
        question, "You are a snarky pirate. Always respond like a pirate."
    )
    response = await chat_completion(msgs, temperature=0.7)
    result("Snarky pirate", response.strip())

    wait_for_user()

    # Step 5: Few-shot + Chain-of-thought
    step(5, "Few-shot and chain-of-thought")
    cot_msgs = with_chain_of_thought(
        "If a train leaves at 3pm traveling 60mph, and another leaves at 4pm going 80mph on the same route, when does the second train catch up?"
    )
    info("Chain-of-thought reasoning:")
    response = await chat_completion(cot_msgs, temperature=0.1)
    info(response.strip()[:300] + "...")

    section("Exercise Complete!", "check")
    success("LLM Basics complete.")
    info("Next: Module 02 — Data Loading (if not done yet)")
    info("Then: Module 06 — Retrieval Strategies")


if __name__ == "__main__":
    asyncio.run(main())
