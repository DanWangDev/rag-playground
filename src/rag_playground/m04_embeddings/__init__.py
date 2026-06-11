"""Module 04: Embeddings — generate and compare vector embeddings."""

from .embed import embed_documents, embed_query
from .similarity import cosine_similarity, find_most_similar

__all__ = ["embed_query", "embed_documents", "cosine_similarity", "find_most_similar"]
