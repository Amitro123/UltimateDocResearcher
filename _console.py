"""
_console.py — Cross-platform console encoding helper.

Call ensure_utf8() at the top of every CLI entry point so that emoji and
other non-ASCII characters don't crash on Windows consoles that default to
cp1252 (or similar narrow encodings).
"""
from __future__ import annotations
import sys


def ensure_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 with replacement for unsupported chars."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
