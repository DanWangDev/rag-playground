# Module 08: Evaluating RAG Quality

## What You'll Learn

- **Retrieval metrics**: Precision@K, Recall@K, MRR, NDCG
- **Generation metrics**: Faithfulness and relevance scoring
- How to build a **test set** of question-answer pairs
- How to **compare** retrieval strategies quantitatively

## Why Evaluation Matters

"You can't improve what you don't measure." RAG systems have two distinct failure modes:

1. **Retrieval failure**: The right documents aren't found
2. **Generation failure**: The right documents are found, but the answer is wrong

Evaluation measures both independently.

## Retrieval Metrics

Given a query, a set of relevant document IDs (ground truth), and retrieved results:

### Precision@K
What fraction of retrieved documents are relevant?
```
Precision@3 = (# relevant in top-3) / 3
```

### Recall@K
What fraction of all relevant documents were retrieved?
```
Recall@3 = (# relevant in top-3) / (total relevant documents)
```

### Mean Reciprocal Rank (MRR)
Where is the first relevant result?
```
MRR = 1 / (rank of first relevant result)
```
If the first relevant is at rank 2: MRR = 1/2 = 0.5

### Normalized Discounted Cumulative Gain (NDCG)
Rewards relevant results at higher ranks:
```
DCG@K = sum(relevance_i / log2(i+1)) for i=1..K
NDCG = DCG / ideal_DCG
```

## Generation Metrics

Exact match isn't useful for evaluating generated text. Instead:

### Faithfulness
Is the generated answer supported by the retrieved context, or is it hallucinated?
LLM-as-judge: "Does this answer only contain information present in the context?"

### Relevance
Does the answer actually address the question asked?
LLM-as-judge: "Does this answer directly address the user's question?"

## What You'll Practice

1. Load a test set of question/answer pairs
2. Run retrieval evaluation (Precision@3, Recall@3, MRR, NDCG)
3. Run generation evaluation (faithfulness, relevance)
4. Compare retrieval strategies on the same test set
