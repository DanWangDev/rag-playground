# Module 04: Embeddings

## What You'll Learn

- What **embeddings** are and how they represent meaning as vectors
- How to **generate** embeddings using a local model (nomic-embed-text via Ollama)
- How **cosine similarity** measures semantic relatedness
- How to **find nearest neighbors** in embedding space
- The relationship between embedding quality and retrieval quality

## Why Embeddings Matter

Computers understand numbers, not words. To search for "climate change causes" in a pile of documents, the computer needs to _compare_ the query to each document. But how do you compare two strings?

**Embeddings** convert text into fixed-length vectors (lists of floats) where semantically similar texts land near each other in vector space:

```
"climate change"       → [0.12, -0.34, 0.56, ..., 0.78]  (768 numbers)
"global warming"       → [0.15, -0.31, 0.52, ..., 0.81]  (very close!)
"renaissance art"      → [-0.42, 0.67, -0.23, ..., -0.15] (far away!)
```

This is the "Retrieval" half of RAG — without embeddings, there's no way to find relevant documents for a query.

## How Embeddings Work

### The Embedding Model

An embedding model is a neural network trained to map text → vectors. Training objective: texts with similar meanings should have vectors close together (high cosine similarity), and unrelated texts should be far apart.

**nomic-embed-text** (our default) produces 768-dimensional vectors. It's trained on a large corpus of text pairs and optimized for retrieval tasks.

### Generating Embeddings

```
Text → Tokenize → Model → Vector (768 floats)
```

Step by step:
1. **Tokenize**: Break text into tokens the model understands
2. **Encode**: Pass tokens through the neural network
3. **Pool**: Combine token-level outputs into a single sentence-level vector
4. **Normalize**: Scale to unit length (for cosine similarity)

Ollama handles steps 1-4 internally. We just send text to `POST /api/embed` and get back a vector.

### Cosine Similarity

The standard measure for embedding similarity:

```
cosine_sim(A, B) = (A · B) / (|A| × |B|)
```

- **1.0** = identical direction (very similar)
- **0.0** = orthogonal (unrelated)
- **-1.0** = opposite direction (rare with modern embeddings)

For normalized vectors (unit length), cosine similarity simplifies to the dot product.

### Batch Embedding

Embedding one text at a time is slow. Most APIs support batch embedding — send multiple texts in one request:

```python
# Slow: N separate API calls
for text in texts:
    vector = await provider.embed_single(model, text)

# Fast: 1 API call
result = await provider.embed(model, texts)
vectors = result.vectors  # list of N vectors
```

## Key Concepts

### Embedding Dimensions

| Dimensions | Example Model | Quality | Speed |
|-----------|--------------|---------|-------|
| 384 | all-MiniLM-L6-v2 | Good | Very fast |
| 768 | nomic-embed-text | Better | Fast |
| 1024 | bge-large-en-v1.5 | Best | Slower |

More dimensions = more capacity to capture nuance, but more memory and slower search.

### Semantic Similarity vs Keyword Matching

| Query | Keyword Match | Embedding Match |
|-------|--------------|-----------------|
| "car" vs "automobile" | No match | High similarity |
| "happy" vs "joyful" | No match | High similarity |
| "bank" (river) vs "bank" (money) | Match! | Low similarity (different meanings) |

Embeddings capture meaning, not just string overlap. This is the fundamental advantage of semantic search.

## Gotchas

1. **Embedding models need to be pulled**: Run `python scripts/pull_models.py` before using this module.
2. **Batch limits**: Ollama has a practical batch limit. Very large batches (>100 texts) may timeout.
3. **Dimension mismatch**: If you change embedding models, you must re-embed all documents. A 768-dim vector can't be compared with a 1024-dim vector.
4. **Normalization matters**: Our provider returns normalized vectors. If you use raw vectors from another source, normalize them before computing cosine similarity.
5. **Language support**: nomic-embed-text is trained primarily on English. For multilingual retrieval, consider bge-m3.

## What You'll Practice

In the exercise, you will:
1. Generate embeddings for individual sentences
2. Compute cosine similarity between pairs
3. Find nearest neighbors in a small collection
4. Visualize a similarity matrix
5. Detect paraphrases using embedding similarity
