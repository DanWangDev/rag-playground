"""Tests for chat completion using mock provider."""

from rag_playground.config.providers.base import ChatMessage, ChatOptions
from tests.helpers.mock_provider import MockProvider


class TestChatCompletion:
    async def test_mock_chat_returns_string(self):
        provider = MockProvider()
        msgs = [ChatMessage(role="user", content="Hello")]
        response = await provider.chat("mock-model", msgs)
        assert isinstance(response, str)
        assert len(response) > 0

    async def test_mock_chat_stream(self):
        provider = MockProvider()
        msgs = [ChatMessage(role="user", content="Hello world")]
        tokens = []
        async for token in provider.chat_stream("mock-model", msgs):
            tokens.append(token)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    async def test_chat_with_options(self):
        provider = MockProvider()
        msgs = [ChatMessage(role="user", content="Test")]
        opts = ChatOptions(temperature=0.0, max_tokens=50)
        response = await provider.chat("mock-model", msgs, opts)
        assert isinstance(response, str)


class TestPromptEngineering:
    def test_with_system_prompt(self):
        from rag_playground.m01_llm_basics.prompt_engineering import with_system_prompt

        msgs = with_system_prompt("Hello", "You are helpful.")
        assert msgs[0].role == "system"
        assert msgs[0].content == "You are helpful."
        assert msgs[1].role == "user"
        assert msgs[1].content == "Hello"

    def test_with_few_shot(self):
        from rag_playground.m01_llm_basics.prompt_engineering import with_few_shot

        examples = [("Input A", "Output A"), ("Input B", "Output B")]
        msgs = with_few_shot("Input C", examples)
        # Should have: user, assistant, user, assistant, user
        assert len(msgs) == 5
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[-1].content == "Input C"

    def test_with_chain_of_thought(self):
        from rag_playground.m01_llm_basics.prompt_engineering import (
            with_chain_of_thought,
        )

        msgs = with_chain_of_thought("What is 2+2?")
        assert len(msgs) == 1
        assert "step by step" in msgs[0].content
        assert "2+2" in msgs[0].content
