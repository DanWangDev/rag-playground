"""Document abstraction and file loaders.

Key Learning Points:
  - Document is a simple dataclass with two fields: page_content + metadata
  - Loaders are async to support future remote sources (S3, HTTP, databases)
  - Encoding detection tries UTF-8 first, falls back to latin-1
  - Markdown loader parses frontmatter (YAML between --- delimiters)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A document loaded from disk (or any source).

    This is the universal currency of a RAG pipeline. Every component
    after data loading works with Document objects.

    Attributes:
        page_content: The full text content of the document.
        metadata: Arbitrary key-value metadata (source, filename, etc.).
    """

    page_content: str
    metadata: dict = field(default_factory=dict)


async def load_text_file(filepath: str | Path) -> Document:
    """Load a plain text (.txt) file as a Document.

    Tries UTF-8 encoding first, falls back to latin-1.

    Args:
        filepath: Path to the .txt file.

    Returns:
        Document with page_content set to file contents and metadata
        including source, filename, and encoding.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        IsADirectoryError: If the path points to a directory.
    """
    path = Path(filepath).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {path}")

    content, encoding = _read_text_with_fallback(path)

    return Document(
        page_content=content,
        metadata={
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix,
            "encoding": encoding,
            "file_size": path.stat().st_size,
        },
    )


async def load_markdown_file(filepath: str | Path) -> Document:
    """Load a markdown (.md) file, parsing optional YAML frontmatter.

    Frontmatter is YAML between --- delimiters at the start of the file:
        ---
        title: My Doc
        tags: [a, b]
        ---
        # Content starts here...

    Frontmatter fields become metadata. The rest is page_content.

    Args:
        filepath: Path to the .md file.

    Returns:
        Document with frontmatter fields merged into metadata.
    """
    path = Path(filepath).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {path}")

    raw, encoding = _read_text_with_fallback(path)

    metadata: dict = {
        "source": str(path),
        "filename": path.name,
        "file_type": ".md",
        "encoding": encoding,
        "file_size": path.stat().st_size,
    }

    # Parse frontmatter if present
    content = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1].strip()
            content = parts[2].strip()
            frontmatter = _parse_simple_yaml(frontmatter_text)
            metadata.update(frontmatter)

    return Document(page_content=content, metadata=metadata)


def _read_text_with_fallback(path: Path) -> tuple[str, str]:
    """Read a file, trying UTF-8 first, then latin-1.

    Returns (content, encoding_used).
    """
    encodings = ["utf-8", "latin-1"]
    for enc in encodings:
        try:
            with open(path, encoding=enc) as f:
                return f.read(), enc
        except UnicodeDecodeError:
            continue
    # Should never reach here (latin-1 decodes everything)
    raise ValueError(f"Could not decode file: {path}")


def _parse_simple_yaml(text: str) -> dict:
    """Parse a minimal subset of YAML for frontmatter.

    Supports:
        key: value         → str value
        key: "quoted"      → str value
        key: [a, b, c]     → list value
        # comment lines    → ignored

    This is intentionally minimal — no PyYAML dependency needed.
    For production RAG, use PyYAML or python-frontmatter.
    """
    result: dict = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Remove surrounding quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            # Parse list: [a, b, c]
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1]
                value = [v.strip().strip('"').strip("'") for v in inner.split(",")]

            result[key] = value
    return result
