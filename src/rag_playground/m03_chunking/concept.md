# Module 03: Text Splitting & Chunking

## What You'll Learn

- Why RAG systems **must** split documents into chunks
- Three splitting strategies: **character**, **recursive**, and **token-based**
- How **chunk overlap** improves retrieval quality
- How to choose chunk size based on your embedding model and LLM
- The trade-off: small chunks vs large chunks

## Why Chunking Matters

LLMs have a **context window** — a maximum number of tokens they can process at once. Modern models handle 8K-128K tokens, but Retrieval-Augmented Generation works best with **focused, relevant context**, not entire documents.

If you embed and retrieve an entire 10-page document for a one-sentence question, you waste tokens, dilute relevance, and increase costs. Chunking breaks documents into smaller, self-contained units:

```
Document (5000 chars)
  ├── Chunk 1 (500 chars): "Machine learning is..."
  ├── Chunk 2 (500 chars): "...supervised learning uses labeled data..."
  ├── Chunk 3 (500 chars): "...unsupervised learning finds patterns..."
  └── Chunk 4 (500 chars): "...deep learning uses neural networks..."
```

Each chunk gets its own embedding. When a user asks about "supervised learning," only chunks 2-3 are retrieved — not the entire document.

## How Chunking Works

### Character Splitting

The simplest strategy: split text every N characters.

```
chunk_size = 100
text = "AAAAABBBBBCCCCCDDDDD"  (20 chars)

Chunk 1: "AAAAA"      (chars 0-9,  10 chars)
Chunk 2: "BBBBB"      (chars 10-19, 10 chars)
```

**Pros:** Simple, fast, deterministic.
**Cons:** Cuts mid-word and mid-sentence. A chunk might end with "The quick brown " and the next starts with "fox jumps over..."

### Recursive Character Splitting

Uses a **separator hierarchy** to split on natural boundaries first:

```
Try "\n\n"  (paragraphs)
  → Try "\n"  (lines)
    → Try ". "  (sentences)
      → Try " "  (words)
        → Fall back to character split
```

This preserves semantic boundaries. A chunk will contain complete paragraphs when possible, complete sentences when not, and only split mid-word as a last resort.

**Pros:** Better semantic coherence per chunk.
**Cons:** Slightly more complex. Chunks can be uneven sizes.

### Token-Aware Splitting

Characters ≠ tokens. A token is roughly 4 characters in English, but can be 1 character (a punctuation mark) or 10+ characters (a long word). The ratio varies by language.

Token-aware splitting counts tokens (approximately) instead of characters, ensuring chunks fit within model context windows.

```
chunk_tokens = 128
text = "The transformer architecture..." (50 tokens)

→ One chunk of ~50 tokens (fits within 128)
```

**Pros:** Model-aware sizing. Prevents context overflow.
**Cons:** Approximation (no tokenizer dependency). Varies by model.

### Chunk Overlap

Overlap creates a sliding window: each chunk shares some text with its neighbors.

```
chunk_size = 100, overlap = 20
text = "AAAAABBBBBCCCCCDDDDDEEEEE"

Chunk 1: "AAAAABBBBBCCCCC"  (100 chars)
Chunk 2: "BBBBBCCCCCDDDDD"  (100 chars, shares "BBBBBCCCCC" with Chunk 1)
Chunk 3: "CCCCCDDDDDEEEEE"  (100 chars, shares "CCCCCDDDDD" with Chunk 2)
```

**Why overlap matters:** A key fact might span the boundary between two non-overlapping chunks. If "Machine learning was invented by Arthur Samuel in 1959" is split into two chunks:

```
Chunk 1: "...Machine learning was invented by"
Chunk 2: "Arthur Samuel in 1959..."
```

Neither chunk alone answers "Who invented machine learning?" With overlap, a chunk captures the full sentence.

## Choosing Chunk Size

| Chunk Size | Best For | Trade-off |
|-----------|----------|-----------|
| Small (128-256) | Factual Q&A, short answers | More chunks, more embeddings, more storage |
| Medium (500-1000) | General RAG (default) | Balanced retrieval granularity |
| Large (2000-4000) | Summarization, long-form answers | Fewer chunks, broader context, less precise retrieval |

## Gotchas

1. **Mid-word splits**: Character splitting on position 73 might cut "transformer" as "transform" + "er." Recursive splitting with word-level separators prevents this.
2. **Token ≠ character**: 500 characters ≈ 125 tokens in English but 250 tokens in Japanese. Token-aware splitting matters for multilingual applications.
3. **Metadata loss**: Each chunk inherits the parent document's metadata. Don't lose the source information when splitting.
4. **Minimum chunk size**: Very small chunks (< 50 chars) often lack useful context. Filter them out.

## What You'll Practice

In the exercise, you will:
1. Split a document with fixed-size character splitting
2. Split the same document with recursive splitting — compare quality
3. Use token-aware splitting — see the character/token difference
4. Experiment with overlap (0, 50, 100) — find mid-chunk splits
5. Compare results on technical documentation vs narrative text
