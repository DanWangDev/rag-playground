"""Retrieval quality metrics.

Key Learning Points:
  - Precision@K: how many of the top-K results are relevant?
  - Recall@K: how many of all relevant docs were found?
  - MRR: where does the first relevant result appear?
  - NDCG: relevance-weighted ranking quality
  - All metrics assume you have ground-truth relevance labels
"""

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant.

    Args:
        retrieved: List of retrieved document IDs (ordered).
        relevant: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-K.

    Args:
        retrieved: List of retrieved document IDs (ordered).
        relevant: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank.

    MRR = 1 / rank_of_first_relevant_result

    Args:
        retrieved: List of retrieved document IDs (ordered).
        relevant: Set of relevant document IDs.

    Returns:
        Score between 0.0 and 1.0. Higher = first relevant result is earlier.
    """
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(
    retrieved: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Rewards highly relevant documents appearing early in results.

    DCG@K = sum(rel_i / log2(i+1)) for i=1..K
    NDCG@K = DCG / ideal_DCG

    Args:
        retrieved: List of retrieved document IDs (ordered).
        relevance_scores: Dict mapping document ID to relevance score (higher = more relevant).
        k: Number of top results to consider.

    Returns:
        Score between 0.0 and 1.0.
    """
    if k <= 0 or not retrieved:
        return 0.0

    # Compute DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / math.log2(i + 1)

    # Compute ideal DCG (sort by relevance descending)
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = 0.0
    for i, rel in enumerate(ideal_rels, 1):
        ideal_dcg += rel / math.log2(i + 1)

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg
