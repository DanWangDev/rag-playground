"""Tests for retrieval evaluation metrics."""

import math

from rag_playground.m08_evaluation.retrieval_metrics import (
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_precision(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_zero_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_no_relevant_docs(self):
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        assert recall_at_k(retrieved, relevant, 3) == 0.0


class TestMRR:
    def test_first_rank(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_rank(self):
        assert math.isclose(mrr(["a", "b", "c"], {"b"}), 0.5)

    def test_no_match(self):
        assert mrr(["a", "b"], {"x"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        scores = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg_score = ndcg(retrieved, scores, 3)
        assert math.isclose(ndcg_score, 1.0, rel_tol=1e-6)
