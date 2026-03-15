"""
new_run.py — Reset the workspace before starting a new research topic.

Solves the "mixed runs" problem where leftover PDFs, stale data files,
and an outdated templates/program.md contaminate the next research cycle.

What it does
------------
1. Archives papers/ → papers/.archive/<timestamp>/   (no deletion — safe)
2. Clears data/ artifacts (all_docs*, train.jsonl, val.jsonl, etc.)
3. If --topic is given, updates templates/program.md with the new topic
4. Prints a summary of every action taken

Usage
-----
    # Before a new research run:
    python new_run.py --topic "retrieval-augmented generation"

    # Just clean without changing the topic:
    python new_run.py

    # Dry-run (see what would happen without doing it):
    python new_run.py --topic "new topic" --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Paths relative to this script's location (= project root)
ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "papers"
DATA_DIR = ROOT / "data"
TEMPLATE_PATH = ROOT / "templates" / "program.md"

# Data files that are regenerated each run — safe to delete
DATA_PATTERNS = [
    "all_docs.txt",
    "all_docs.txt.bak",
    "all_docs_cleaned.txt",
    "all_docs_cleaned.txt.bak",
    "corpus_report.json",
    "corpus_report.json.bak",
    "metadata.jsonl",
    "train.jsonl",
    "val.jsonl",
]

PROMPT_CACHE_DB   = ROOT / "dashboard" / "cache" / "prompts.db"
PROMPT_CACHE_JSONL = ROOT / "dashboard" / "cache" / "prompts.jsonl"  # legacy


def _archive_papers(dry_run: bool) -> None:
    """Move all files in papers/ into a timestamped archive subdirectory."""
    if not PAPERS_DIR.exists():
        print(f"  [skip] {PAPERS_DIR} does not exist — nothing to archive.")
        return

    files = [f for f in PAPERS_DIR.iterdir() if f.is_file()]
    if not files:
        print(f"  [skip] {PAPERS_DIR} is already empty.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = PAPERS_DIR / ".archive" / ts

    print(f"\n📦  Archiving {len(files)} file(s) from papers/ → papers/.archive/{ts}/")
    for f in files:
        dest = archive_dir / f.name
        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dest)
        print(f"    {'(dry) ' if dry_run else ''}moved: {f.name}")


def _clear_data(dry_run: bool) -> None:
    """Delete stale data artifacts in data/."""
    if not DATA_DIR.exists():
        print(f"\n  [skip] {DATA_DIR} does not exist — nothing to clear.")
        return

    print(f"\n🗑️   Clearing stale data/ artifacts:")
    cleared = 0
    for name in DATA_PATTERNS:
        path = DATA_DIR / name
        if path.exists():
            print(f"    {'(dry) ' if dry_run else ''}delete: {path.name}")
            if not dry_run:
                path.unlink()
            cleared += 1

    if cleared == 0:
        print("    (nothing to clear)")


def _clear_prompt_cache(dry_run: bool) -> None:
    """Delete the SQLite prompt cache so stale responses don't pollute the new run."""
    cleared_any = False

    # Primary: SQLite cache (current implementation)
    if PROMPT_CACHE_DB.exists():
        size = PROMPT_CACHE_DB.stat().st_size
        print(f"\n🧹  Clearing prompt cache ({size / 1024:.1f} KB): {PROMPT_CACHE_DB.name}")
        if not dry_run:
            PROMPT_CACHE_DB.unlink()
        else:
            print(f"    (dry) would delete {PROMPT_CACHE_DB}")
        cleared_any = True

    # Legacy: JSONL file left over from pre-SQLite versions
    if PROMPT_CACHE_JSONL.exists():
        size = PROMPT_CACHE_JSONL.stat().st_size
        print(f"    Also removing legacy JSONL cache ({size / 1024:.1f} KB): {PROMPT_CACHE_JSONL.name}")
        if not dry_run:
            PROMPT_CACHE_JSONL.unlink()
        else:
            print(f"    (dry) would delete {PROMPT_CACHE_JSONL}")
        cleared_any = True

    if not cleared_any:
        print(f"\n  [skip] No prompt cache found — nothing to clear.")


def _update_template(topic: str, dry_run: bool) -> None:
    """Overwrite the Topic line in templates/program.md."""
    if not TEMPLATE_PATH.exists():
        print(f"\n  [skip] {TEMPLATE_PATH} not found — skipping topic update.")
        return

    text = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Replace the line after "## Topic"
    new_text, n = re.subn(
        r"(## Topic\s*\n)([^\n]+)",
        rf"\g<1>{topic}",
        text,
    )

    if n == 0:
        print(f"\n  [warn] Could not find '## Topic' section in {TEMPLATE_PATH} — skipping.")
        return

    print(f"\n📝  Updating templates/program.md:")
    print(f"    topic → {topic!r}")
    if not dry_run:
        TEMPLATE_PATH.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset workspace before a new research run."
    )
    parser.add_argument(
        "--topic", default=None,
        help="New research topic to write into templates/program.md",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making any changes",
    )
    parser.add_argument(
        "--skip-archive", action="store_true",
        help="Skip archiving papers/ (use if you already cleaned it manually)",
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip clearing data/ artifacts",
    )
    parser.add_argument(
        "--keep-cache", action="store_true",
        help=(
            "Keep the LLM prompt cache (dashboard/cache/prompts.jsonl). "
            "Useful when retrying a failed run to avoid redundant API calls."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        print("⚠️  DRY RUN — no changes will be made.\n")

    if not args.skip_archive:
        _archive_papers(args.dry_run)

    if not args.skip_data:
        _clear_data(args.dry_run)

    if not args.keep_cache:
        _clear_prompt_cache(args.dry_run)
    else:
        print("\n  [skip] Keeping prompt cache (--keep-cache set).")

    if args.topic:
        _update_template(args.topic, args.dry_run)
    else:
        print(
            "\n💡  Tip: pass --topic \"your new topic\" to also update templates/program.md."
        )

    print(
        "\n✅  Workspace reset complete."
        + (" (dry run — nothing changed)" if args.dry_run else "")
        + "\n    You can now run the collector:\n"
        + "    python -m collector.ultimate_collector --pdf-dir papers/ --output-dir data/"
    )


if __name__ == "__main__":
    main()
