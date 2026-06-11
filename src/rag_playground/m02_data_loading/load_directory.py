"""Recursive directory loader.

Key Learning Points:
  - Recursive directory traversal with glob filtering
  - Each file is dispatched to the appropriate loader by extension
  - Metadata is enriched with directory structure information
  - Batch loading is the first step toward indexing a knowledge base
"""

import fnmatch
from pathlib import Path

from .document import Document, load_markdown_file, load_text_file


async def load_directory(
    dirpath: str | Path,
    glob_pattern: str = "**/*.{md,txt}",
    recursive: bool = True,
) -> list[Document]:
    """Load all matching documents from a directory tree.

    Args:
        dirpath: Root directory to scan.
        glob_pattern: File matching pattern (e.g., "**/*.md" for all markdown).
        recursive: If True, descend into subdirectories.

    Returns:
        List of Document objects, one per matched file.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        NotADirectoryError: If the path is not a directory.
    """
    path = Path(dirpath).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")

    # Collect all matching file paths
    files: list[Path] = []
    if recursive:
        for ext in ["*.md", "*.txt"]:
            files.extend(path.rglob(ext))
    else:
        for ext in ["*.md", "*.txt"]:
            files.extend(path.glob(ext))

    # Apply glob filter for additional filtering
    if glob_pattern and glob_pattern not in ("**/*.{md,txt}", "*"):
        pattern_parts = glob_pattern.split("/")
        base_pattern = pattern_parts[-1]
        filtered: list[Path] = []
        for f in files:
            if fnmatch.fnmatch(f.name, base_pattern):
                filtered.append(f)
        files = filtered

    # Sort for deterministic order
    files = sorted(files)

    # Load each file
    documents: list[Document] = []
    for filepath in files:
        suffix = filepath.suffix.lower()
        try:
            if suffix == ".md":
                doc = await load_markdown_file(filepath)
            elif suffix == ".txt":
                doc = await load_text_file(filepath)
            else:
                continue  # Skip unsupported types

            # Enrich metadata with directory context
            rel_path = str(filepath.relative_to(path))
            doc.metadata["relative_path"] = rel_path
            doc.metadata["parent_dir"] = str(filepath.parent.relative_to(path))

            # top_level_dir is the first path component, or "." if at root
            parts = rel_path.replace("\\", "/").split("/")
            doc.metadata["top_level_dir"] = parts[0] if len(parts) > 1 else "."

            documents.append(doc)
        except Exception:
            # Skip files that fail to load (permission errors, etc.)
            continue

    return documents
