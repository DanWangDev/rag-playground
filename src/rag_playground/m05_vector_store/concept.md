# Module 05: Building a Vector Store

## What You'll Learn

- How to build an **in-memory vector store** from scratch
- How **k-NN search** finds the most similar vectors
- How **metadata filtering** narrows search results
- How **persistence** saves and loads vector stores to/from disk

## Why Vector Stores Matter

Embeddings without a store are just numbers. To retrieve relevant documents for a query, you need to:

1. **Store** the embeddings with their associated chunks and metadata
2. **Search** for the most similar embeddings to a query vector
3. **Return** the chunks corresponding to those embeddings

This is the "R" in RAG — retrieval. The vector store is the engine that makes retrieval possible.

## How an In-Memory Vector Store Works

### Data Structure

```
InMemoryVectorStore
├── vectors: list[list[float]]     ← Embedding vectors
├── chunks: list[Chunk]            ← Associated chunks
├── metadata: list[dict]           ← Per-entry metadata
└── index: dict[str, int]          ← chunk_id → position lookup
```

All lists are parallel: `vectors[i]` corresponds to `chunks[i]` and `metadata[i]`.

### Adding Entries

```python
store = InMemoryVectorStore(embedding_model="nomic-embed-text", dimensions=768)
store.add(embedding, chunk, metadata)
```

### Searching (Brute-Force k-NN)

Given a query vector, the store:
1. Computes cosine similarity with every stored vector
2. Sorts by similarity (descending)
3. Returns the top-k results

Complexity: O(n × d) where n = number of vectors, d = dimensions.
For 768-dim vectors and 1000 entries: ~768K operations — fine in-memory.
For 1M entries: ~768M operations — you'd want an approximate index (ANN).

### Metadata Filtering

Pre-filter: only search vectors matching certain metadata fields.
Post-filter: search all vectors, then filter results by metadata.

Our implementation uses post-filtering — simpler, and fine for small collections.

### Persistence

Save to JSON:
```python
store.save_to_file("my_store.vectorstore.json")
```

Load from JSON:
```python
store = InMemoryVectorStore.load_from_file("my_store.vectorstore.json")
```

## Key Concepts

### Brute-Force vs Approximate Search

| Method | Accuracy | Speed | Use Case |
|--------|----------|-------|----------|
| Brute-force k-NN | 100% | O(n×d) | Small collections (<10K) |
| HNSW (approximate) | ~99% | O(log n) | Large collections (>10K) |
| IVF (approximate) | ~95% | O(log n) | Very large collections (>1M) |

We implement brute-force. It's exact, simple to understand, and fast enough for learning.

### Similarity Threshold

Filter results to only return chunks above a minimum similarity score:

```python
results = store.search(query_vector, top_k=5, min_score=0.7)
```

This prevents the retriever from returning irrelevant results when the knowledge base doesn't contain relevant information.

## Gotchas

1. **Dimension mismatch**: Adding a 384-dim vector to a 768-dim store raises an error. The store is initialized with a fixed dimension.
2. **Memory usage**: Each 768-dim float vector is ~3KB. 10K vectors ≈ 30MB. Monitor memory for large collections.
3. **No concurrency**: InMemoryVectorStore is not thread-safe by default. Add locking for multi-threaded use.
4. **JSON precision**: Saving floats to JSON may lose tiny amounts of precision. For production, use a binary format.
