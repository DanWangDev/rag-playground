# Module 06: Retrieval Strategies

## What You'll Learn

- **Basic retrieval**: embed query → search vector store → return top-k
- **Query preprocessing**: expand, decompose, and HyDE queries for better recall
- **Multi-query retrieval**: generate variants, search in parallel, fuse results
- **LLM reranking**: use the LLM itself to score and reorder retrieved candidates

## Why Retrieval Strategies Matter

The "R" in RAG. A naive retrieval might miss relevant documents because:
- The query uses different words than the documents (vocabulary mismatch)
- The query is too broad or too narrow
- Multiple relevant facts are scattered across different documents

Advanced retrieval strategies address each of these failure modes.

## Strategies

### Basic Retrieval
Embed the query, search the vector store, return top-k. Simple and fast.

### Query Expansion
Use the LLM to expand the query with synonyms and related terms:
```
"ML" → "machine learning, artificial intelligence, deep learning, neural networks"
```

### Query Decomposition
Break complex questions into simpler sub-questions:
```
"Compare ML and quantum computing" →
  Q1: "What is machine learning?"
  Q2: "What is quantum computing?"
  Q3: "How do ML and quantum computing relate?"
```

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer, then embed THAT as the query:
```
Q: "What causes climate change?"
Hypothetical: "Climate change is primarily caused by greenhouse gas emissions..."
Embed the hypothetical → search for real documents similar to the hypothetical
```

### Multi-Query Retrieval
Generate N query variants, retrieve for each, fuse results with Reciprocal Rank Fusion (RRF):
```
RRF(score) = sum over queries: 1 / (k + rank)
```

### LLM Reranking
Retrieve top-N candidates (e.g., 20), then ask the LLM to score their relevance:
```
"On a scale of 1-10, how relevant is this passage to the query '...'?"
```

## Gotchas
- Multi-query is slower (N API calls) but has better recall
- Reranking adds latency but improves precision
- HyDE works well for fact-seeking queries but poorly for creative ones
