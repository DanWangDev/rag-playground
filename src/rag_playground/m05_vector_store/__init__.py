"""Module 05: Vector Store — store and search embeddings."""

from .in_memory_store import InMemoryVectorStore
from .persistence import load_from_file, save_to_file

__all__ = ["InMemoryVectorStore", "save_to_file", "load_from_file"]
