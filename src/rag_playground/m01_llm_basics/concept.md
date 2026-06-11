# Module 01: LLM Basics

## What You'll Learn

- How to chat with a **local LLM** via Ollama
- The difference between **streaming** and **non-streaming** responses
- How **temperature** controls randomness vs determinism
- Basic **prompt engineering**: system prompts, few-shot, chain-of-thought

## Why Local LLMs Matter

Cloud APIs (OpenAI, Anthropic) are convenient but:
- Cost money per token
- Send your data to external servers
- Require internet connectivity
- Have rate limits and usage policies

Local models via Ollama are free, private, offline-capable, and unlimited. For learning RAG, local models mean you can experiment without worrying about API bills.

## How Chat Completions Work

```
User: "What is machine learning?"
  → System prompt (optional): "You are a helpful assistant."
  → Model processes tokens
  → Assistant: "Machine learning is a field of AI..."
```

### Non-Streaming (chat)

```python
response = await provider.chat(model, messages)
# Returns the full response at once
```

### Streaming (chat_stream)

```python
async for token in provider.chat_stream(model, messages):
    print(token, end="")  # Tokens appear one at a time
```

Streaming gives users immediate feedback — important for good UX in chat applications.

### Temperature

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Deterministic, picks most likely token | Factual Q&A, code generation |
| 0.5 | Balanced | General conversation |
| 1.0 | Creative, diverse outputs | Brainstorming, creative writing |

Lower temperature = more focused, fewer hallucinations (good for RAG).

### Prompt Engineering

**System Prompts** set the assistant's behavior:
```
System: "Answer questions using ONLY the provided context. If the answer
        isn't in the context, say 'I don't have that information.'"
```

**Few-Shot Prompting** provides examples:
```
User: "Classify: 'I love this product'"
Assistant: "Positive"
User: "Classify: 'Terrible experience'"
Assistant: "Negative"
User: "Classify: 'It works fine'"
Assistant: "Neutral"
```

**Chain-of-Thought** asks the model to reason step-by-step:
```
"Let's think about this step by step. First..."
```

## Gotchas

1. **Model must be pulled first**: Run `python scripts/pull_models.py`
2. **Memory usage**: Qwen 7B needs ~5-7GB RAM. Ensure Ollama has enough.
3. **Context window**: Qwen 2.5 7B supports 32K tokens. Long conversations need context management.
4. **Temperature 0 ≠ always same answer**: Local models with temperature 0 may still vary slightly.
