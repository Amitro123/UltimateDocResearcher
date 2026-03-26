"""
autoresearch/incremental_collect.py
-------------------------------------
Incremental document collector for UltimateDocResearcher.

Tracks previously collected sources via `data/collect_metadata.jsonl`.
Each entry records a `file_hash` (for local files) or `url_hash` (for
remote sources) and `last_modified` timestamp.  On subsequent runs:
  - Unchanged local files are skipped (same mtime + size hash).
  - Previously fetched URLs are skipped (url_hash already present).
  - New or changed sources are collected and appended to all_docs.txt.

This avoids re-processing the same content on every run and makes the
pipeline safe to call repeatedly as new papers/URLs are added.

Usage:
    from autoresearch.incremental_collect import IncrementalCollector

    ic = IncrementalCollector(data_dir="data/", pdf_dir="papers/")
    new_docs, skipped = ic.run(topic="RAG architecture")
    print(f"Collected {new_docs} new docs, skipped {skipped} unchanged.")

Or as a CLI:
    python -m autoresearch.incremental_collect \\
        --topic "RAG architecture" \\
        --pdf-dir papers/ \\
        --data-dir data/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_METADATA_FILE = "collect_metadata.jsonl"
_DOCS_FILE = "all_docs.txt"

# PDF / text extensions treated as local sources
_LOCAL_EXTENSIONS = {".pdf", ".txt", ".md", ".rst"}


# ── Hash helpers ─────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """SHA1 of file content (first 512 KB for speed on large PDFs)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(512 * 1024))
    return h.hexdigest()[:16]


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def _path_hash(path: Path) -> str:
    """Combines mtime + size so unchanged files are quickly detected."""
    try:
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return _file_hash(path)


# ── Metadata store ────────────────────────────────────────────────────────────

class CollectMetadata:
    """Simple JSONL-backed key/value store for source tracking."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / _METADATA_FILE
        self._store: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry.get("key")
                    if key:
                        self._store[key] = entry
                except json.JSONDecodeError:
                    pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            for entry in self._store.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def seen(self, key: str, current_hash: str) -> bool:
        """Return True if we already have this key with the same hash."""
        entry = self._store.get(key)
        if not entry:
            return False
        return entry.get("hash") == current_hash

    def mark(self, key: str, current_hash: str, **extra) -> None:
        self._store[key] = {
            "key": key,
            "hash": current_hash,
            "last_seen": time.time(),
            **extra,
        }

    def flush(self) -> None:
        self._save()

    @property
    def total(self) -> int:
        return len(self._store)


# ── Incremental collector ────────────────────────────────────────────────────

class IncrementalCollector:
    """
    Wraps UltimateCollector with hash-based deduplication so only new or
    changed sources are fetched and appended to data/all_docs.txt.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/",
        pdf_dir: str | Path = "papers/",
        verbose: bool = True,
    ) -> None:
        self.data_dir = ROOT / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
        self.pdf_dir = ROOT / pdf_dir if not Path(pdf_dir).is_absolute() else Path(pdf_dir)
        self.verbose = verbose
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._meta = CollectMetadata(self.data_dir)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[incremental] {msg}")

    # ── Local files ────────────────────────────────────────────────────────

    def _new_local_files(self) -> list[Path]:
        """Return local files that are new or changed since last run."""
        if not self.pdf_dir.exists():
            return []
        new = []
        for p in self.pdf_dir.rglob("*"):
            if p.suffix.lower() not in _LOCAL_EXTENSIONS or not p.is_file():
                continue
            key = f"file:{p.resolve()}"
            current_hash = _path_hash(p)
            if self._meta.seen(key, current_hash):
                self._log(f"  SKIP (unchanged) {p.name}")
            else:
                new.append(p)
                self._meta.mark(key, current_hash, path=str(p))
        return new

    # ── Remote URLs ────────────────────────────────────────────────────────

    def _filter_urls(self, urls: list[str]) -> list[str]:
        """Return only URLs not yet collected."""
        new = []
        for url in urls:
            key = f"url:{url}"
            h = _url_hash(url)
            if self._meta.seen(key, h):
                self._log(f"  SKIP (cached)   {url[:80]}")
            else:
                new.append(url)
                self._meta.mark(key, h, url=url)
        return new

    # ── Main entry ─────────────────────────────────────────────────────────

    def run(
        self,
        topic: str = "",
        extra_urls: list[str] | None = None,
        github_repos: list[str] | None = None,
        queries: list[str] | None = None,
        force: bool = False,
    ) -> tuple[int, int]:
        """
        Run incremental collection.

        Returns (new_doc_count, skipped_count).
        If *force* is True, ignore cached hashes and re-collect everything.
        """
        if force:
            self._log("Force mode: ignoring cached hashes")
            self._meta._store.clear()

        new_files = self._new_local_files()
        new_urls = self._filter_urls(extra_urls or [])

        # GitHub repos: treat each as a URL-keyed source
        new_repos: list[str] = []
        for repo in (github_repos or []):
            key = f"github:{repo}"
            h = _url_hash(repo)
            if not force and self._meta.seen(key, h):
                self._log(f"  SKIP (cached)   github:{repo}")
            else:
                new_repos.append(repo)
                self._meta.mark(key, h, repo=repo)

        skipped = (
            self._meta.total
            - len(new_files)
            - len(new_urls)
            - len(new_repos)
        )
        skipped = max(skipped, 0)

        total_new = len(new_files) + len(new_urls) + len(new_repos)

        if total_new == 0 and not queries:
            self._log(
                f"Nothing new to collect "
                f"(metadata tracks {self._meta.total} sources). "
                "Use --force to re-collect everything."
            )
            self._meta.flush()
            return 0, skipped

        self._log(
            f"New: {len(new_files)} local files, "
            f"{len(new_urls)} URLs, "
            f"{len(new_repos)} GitHub repos"
            + (f", {len(queries or [])} search queries" if queries else "")
        )

        # Delegate actual collection to UltimateCollector
        new_docs = self._collect_new(
            topic=topic,
            new_files=new_files,
            new_urls=new_urls,
            new_repos=new_repos,
            queries=queries or [],
        )

        self._meta.flush()
        return new_docs, skipped

    def _collect_new(
        self,
        topic: str,
        new_files: list[Path],
        new_urls: list[str],
        new_repos: list[str],
        queries: list[str],
    ) -> int:
        """Run UltimateCollector only for new sources, append to all_docs.txt."""
        try:
            from collector.ultimate_collector import UltimateCollector
        except ImportError as e:
            self._log(f"⚠️  Could not import UltimateCollector: {e}")
            return 0

        collector = UltimateCollector(
            pdf_paths=[str(self.pdf_dir)] if new_files else None,
            google_queries=queries,
            urls=new_urls,
            github_repos=new_repos,
            output_dir=str(self.data_dir),
        )

        # Capture docs before running so we know the delta
        docs_path = self.data_dir / _DOCS_FILE
        before_size = docs_path.stat().st_size if docs_path.exists() else 0

        try:
            collector.run()
        except Exception as exc:
            self._log(f"⚠️  Collector error: {exc} — continuing with partial results")

        after_size = docs_path.stat().st_size if docs_path.exists() else 0
        added_bytes = after_size - before_size

        self._log(
            f"Appended ~{added_bytes / 1024:.1f} KB to {docs_path.relative_to(ROOT)}"
        )

        # Rough doc count estimate from line count delta
        if added_bytes > 0:
            try:
                # Count non-empty lines added (each doc is one or more lines)
                content = docs_path.read_text(encoding="utf-8", errors="replace")
                new_doc_estimate = max(1, content[before_size:].count("\n\n"))
                return new_doc_estimate
            except Exception:
                return 1
        return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> int:
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        try:
            _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Incremental document collection for UltimateDocResearcher",
    )
    parser.add_argument("--topic",     default="",      help="Research topic (optional)")
    parser.add_argument("--pdf-dir",   default="papers/", help="Local PDF/text directory")
    parser.add_argument("--data-dir",  default="data/",   help="Output data directory")
    parser.add_argument("--urls",      nargs="*", default=[], help="Extra URLs to collect")
    parser.add_argument("--github",    nargs="*", default=[], help="GitHub repos (user/repo)")
    parser.add_argument("--queries",   nargs="*", default=[], help="Web search queries")
    parser.add_argument("--force",     action="store_true",   help="Re-collect all sources")
    parser.add_argument("--verbose",   action="store_true",   help="Verbose output")
    args = parser.parse_args()

    ic = IncrementalCollector(
        data_dir=args.data_dir,
        pdf_dir=args.pdf_dir,
        verbose=True,
    )
    new, skipped = ic.run(
        topic=args.topic,
        extra_urls=args.urls,
        github_repos=args.github,
        queries=args.queries,
        force=args.force,
    )
    print(f"\n[incremental] Done. New docs: {new}, Skipped (cached): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
