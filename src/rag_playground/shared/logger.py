"""Structured console logger using Rich.

Mirrors dynamodb-playground's logger pattern: emoji-prefixed sections,
numbered steps, and color-coded output levels.

Key Learning Points:
  - Rich Console provides markup, panels, tables beyond print()
  - Structured logging makes CLI exercises readable and scannable
  - Consistent emoji prefixes create a visual language for the user
"""

import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(file=sys.stderr, highlight=False)

# Emoji constants for consistent visual language
EMOJI = {
    "rocket": "🚀",
    "check": "✅",
    "cross": "❌",
    "warn": "⚠️",
    "info": "ℹ️",
    "gear": "⚙️",
    "books": "📚",
    "search": "🔍",
    "brain": "🧠",
    "puzzle": "🧩",
    "link": "🔗",
    "sparkles": "✨",
    "chart": "📊",
    "file": "📄",
    "folder": "📁",
    "db": "🗄️",
    "chip": "💾",
    "test": "🧪",
    "chat": "💬",
    "robot": "🤖",
    "clock": "⏱️",
}


def section(title: str, emoji: str = "info") -> None:
    """Print a major section header."""
    icon = EMOJI.get(emoji, emoji)
    console.print()
    console.print(Panel(f"{icon}  {title}", style="bold cyan"))
    console.print()


def step(number: int, description: str) -> None:
    """Print a numbered step in an exercise."""
    console.print(f"  [bold yellow]Step {number}:[/bold yellow] {description}")


def info(message: str) -> None:
    """Print an informational message."""
    console.print(f"  {EMOJI['info']}  {message}")


def success(message: str) -> None:
    """Print a success message."""
    console.print(f"  {EMOJI['check']}  [green]{message}[/green]")


def warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"  {EMOJI['warn']}  [yellow]{message}[/yellow]")


def error(message: str) -> None:
    """Print an error message."""
    console.print(f"  {EMOJI['cross']}  [red]{message}[/red]")


def result(label: str, value: Any) -> None:
    """Print a labeled result value."""
    console.print(f"  [bold]{label}:[/bold] {value}")


def code_block(code: str, language: str = "python") -> None:
    """Print a syntax-highlighted code block."""
    from rich.syntax import Syntax

    syntax = Syntax(code, language, theme="monokai", line_numbers=False)
    console.print(syntax)


def table(headers: list[str], rows: list[list[Any]], title: str = "") -> None:
    """Print a formatted table."""
    t = Table(title=title, header_style="bold cyan")
    for h in headers:
        t.add_column(h)
    for row in rows:
        t.add_row(*[str(c) for c in row])
    console.print(t)


def divider() -> None:
    """Print a horizontal divider."""
    console.print("─" * 60, style="dim")
