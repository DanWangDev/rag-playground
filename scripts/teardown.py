"""Clean up RAG playground artifacts.

Usage:
    python scripts/teardown.py
    python scripts/teardown.py --force   # Skip confirmation

Key Learning Points:
  - Teardown scripts clean up generated artifacts, not source code
  - Confirmation prompts prevent accidental data loss
  - Only removes cache/index files, never touches data/ or src/
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up RAG playground artifacts")
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove downloaded models and virtual environment",
    )
    args = parser.parse_args()

    if not args.force:
        response = input(
            "This will remove cached vector stores and indices. Continue? [y/N]: "
        )
        if not response.lower().startswith("y"):
            print("Aborted.")
            return

    removed = 0

    # Remove vector store cache files
    for root, _dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".vectorstore.json"):
                path = os.path.join(root, f)
                os.remove(path)
                print(f"  Removed: {path}")
                removed += 1

    # Remove pytest cache
    for cache_dir in [".pytest_cache", "__pycache__"]:
        for root, dirs, _files in os.walk("."):
            if cache_dir in dirs:
                import shutil

                path = os.path.join(root, cache_dir)
                shutil.rmtree(path, ignore_errors=True)
                print(f"  Removed: {path}")
                removed += 1

    # Remove coverage artifacts
    for cov_file in [".coverage", "coverage.xml"]:
        if os.path.exists(cov_file):
            os.remove(cov_file)
            print(f"  Removed: {cov_file}")
            removed += 1

    if removed == 0:
        print("Nothing to clean up.")
    else:
        print(f"\nCleaned up {removed} artifact(s).")


if __name__ == "__main__":
    main()
