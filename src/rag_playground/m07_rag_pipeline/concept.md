# Module 07: RAG Pipeline

## What You'll Learn

- How to assemble the **complete RAG pipeline**: load → chunk → embed → store → retrieve → augment → generate
- How to build **RAG prompt templates** that constrain the LLM to use context
- How to handle **streaming** RAG responses
- Common **failure modes** and how to detect them

## How RAG Works End-to-End

```
1. Documents loaded (Module 02)
2. Documents chunked (Module 03)
3. Chunks embedded (Module 04)
4. Embeddings stored in vector store (Module 05)
5. User asks a question
6. Question embedded → search store → retrieve top-k chunks (Module 06)
7. Retrieved chunks + question → formatted prompt
8. Prompt → LLM → answer with citations (Module 01)
```

### The Augment Step

This is where RAG differs from plain LLM chat. The retrieved chunks are formatted into the prompt:

```
System: You are a helpful assistant. Answer the question using ONLY
        the provided context. If the answer is not in the context,
        say "I don't have enough information to answer that."

Context:
[Chunk 1] Machine learning is a field of AI...
[Chunk 2] Supervised learning uses labeled data...
[Chunk 3] Deep learning uses neural networks...

Question: What is machine learning?

Answer:
```

### Why the System Prompt Matters

Without instructions to use ONLY context, the LLM may:
- Ignore retrieved chunks and use its training data
- Confidently answer questions not covered by the documents
- Hallucinate sources

A strong RAG system prompt is essential for grounded answers.

### Streaming RAG

The retrieval step is fast (~10ms). The generation step is slow (~seconds).
Streaming sends tokens as they're generated, showing the answer build in real-time.

## Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Irrelevant retrieval | Answer doesn't address the question | Better chunking, reranking |
| Context overflow | LLM ignores parts of context | Smaller chunks, summarization |
| Hallucination | Plausible but wrong answer | Stricter system prompt, citations |
| Out-of-domain | LLM answers about topics not in docs | System prompt: "say I don't know" |
