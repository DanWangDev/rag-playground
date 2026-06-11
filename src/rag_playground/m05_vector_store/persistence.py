"""JSON file persistence for the vector store.

Key Learning Points:
  - JSON is human-readable and debuggable — great for learning
  - Float precision: JSON may lose ~1e-15 per value (negligible for similarity)
  - For production, use a binary format (Parquet, Lance) or a vector DB
"""

import json
from pathlib import Path

from .in_memory_store import InMemoryVectorStore


def save_to_file(store: InMemoryVectorStore, filepath: str | Path) -> None:
    """Save a vector store to a JSON file.

    The file is human-readable but can be large for big stores.
    A 10K-entry, 768-dim store is ~30MB of JSON.

    Args:
        store: The InMemoryVectorStore to save.
        filepath: Destination file path (.json extension recommended).
    """
    path = Path(filepath)
    data = store.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_from_file(filepath: str | Path) -> InMemoryVectorStore:
    """Load a vector store from a JSON file.

    Args:
        filepath: Path to the saved .json file.

    Returns:
        A new InMemoryVectorStore populated with the saved data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Vector store file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return InMemoryVectorStore.from_dict(data)
