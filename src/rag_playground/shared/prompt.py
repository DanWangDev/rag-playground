"""Step-by-step pause mechanism for interactive exercises.

Mirrors dynamodb-playground's prompt.ts: pauses between steps so the
user can read output before continuing.

Key Learning Points:
  - Interactive CLI exercises need pacing — don't dump everything at once
  - A simple "press Enter" prompt separates explanation from execution
  - --no-pause flag enables automated/smoke-test runs
"""

import os
import sys


def wait_for_user(message: str = "Press Enter to continue...") -> None:
    """Pause execution until the user presses Enter.

    Set NO_PAUSE=1 in environment to skip all pauses (for automated runs).
    """
    if os.environ.get("NO_PAUSE"):
        return
    try:
        input(f"\n  {message}")
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Ask a yes/no question. Returns True for 'yes'."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(f"\n  {question}{suffix}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if not answer:
        return default
    return answer.startswith("y")
