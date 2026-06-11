"""Module 02: Data Loading — read documents from files into Document objects."""

from .document import Document, load_text_file, load_markdown_file
from .load_directory import load_directory

__all__ = ["Document", "load_text_file", "load_markdown_file", "load_directory"]
